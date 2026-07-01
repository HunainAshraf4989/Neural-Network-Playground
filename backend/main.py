"""Process entry point — constructs the single ``ArchitectureStore`` and runs the
chosen server mode (spec §11; CLAUDE.md invariant 2: one store, one mutation path).

``MODE`` env var:
  (default)   full mode: MCP stdio loop **and** the uvicorn/websocket server on
              :8765, sharing one store and one broadcast path. This is what
              Claude Desktop launches.
  standalone  websocket-only server (frontend dev without Claude Desktop).

**Single canvas owner.** The whole point of the project is that a human *watches*
the agent build — so there must be exactly one live canvas. Whenever a fresh
backend starts (Claude Desktop respawns the MCP server, the user starts a dev
server, ...), it reclaims the websocket port from any *previous nn-architect
backend* still holding it (``_reclaim_canvas_port``): it terminates that prior
instance and takes over, so the browser — which auto-reconnects — re-attaches to
the new, authoritative store and the human keeps seeing live edits. This is the
fix for the "many stale backends → new MCP calls run headless → human sees
nothing" failure. The reclaim is *safe*: it only ever kills a process it can
positively identify as our own backend (a per-port pidfile + ``/proc`` cmdline
check); an unrelated service on the port is never touched — for that we keep the
old headless+``canvas_warning`` fallback.

stdout is reserved for MCP protocol framing (invariant 1). All logging is forced
onto stderr **and** a rotating log file (``LOG_FILE``) so no library can leak a
line onto stdout, and so every request can be replayed for debugging. uvicorn is
given ``log_config=None`` for the same reason: its default config writes the
access log to *stdout*, which would corrupt the MCP stream in full mode — with no
config it propagates to the root logger (stderr + file) instead.
"""

import asyncio
import atexit
import logging
import logging.handlers
import os
import signal
import socket
import sys
import tempfile
import time
from pathlib import Path

import uvicorn

from mcp_tools import build_mcp
from store import ArchitectureStore
from ws_app import build_app

log = logging.getLogger("nn_architect.main")

_REPO_ROOT = Path(__file__).resolve().parent.parent
WS_HOST = "0.0.0.0"
WS_PORT = int(os.environ.get("WS_PORT", "8765"))


def _configure_logging() -> None:
    """Send all logs to stderr and to a rotating file; never to stdout."""
    level = os.environ.get("LOG_LEVEL", "INFO")
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)
    handlers = [stderr_handler]

    log_file = os.environ.get("LOG_FILE", str(_REPO_ROOT / "logs" / "nn_architect.log"))
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(fmt)
        handlers.append(file_handler)

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)
    log.info("logging to stderr + %s", log_file or "(file disabled)")


def _ws_server(store):
    """Build the websocket app + uvicorn server; return ``(server, broadcast)``."""
    app, broadcast = build_app(store)
    config = uvicorn.Config(app, host=WS_HOST, port=WS_PORT, log_config=None)
    return uvicorn.Server(config), broadcast


def _port_available(host, port) -> bool:
    """True if (host, port) can be bound right now. Used as a pre-flight check so
    we *never* start uvicorn on a busy port: uvicorn's bind-failure path runs on
    the same event loop as the MCP stdio server and wedges it (the agent then
    sees "failed to connect" with no clue why). The stdio channel is the agent's
    only contract; the websocket/canvas is an optional human convenience, so when
    the port is taken we skip the websocket and run MCP-only.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
            return True
        except OSError:
            return False


# -- single canvas owner: reclaim the port from our own stale backend ----------
#
# A pidfile in the system temp dir, keyed by WS_PORT, records which process
# currently owns the canvas. A new backend uses it to recognise — and reclaim the
# port from — a *previous nn-architect backend*, while never touching an
# unrelated service that merely happens to hold the port.

def _pidfile_path() -> Path:
    return Path(tempfile.gettempdir()) / f"nn_architect_canvas_{WS_PORT}.pid"


def _read_pidfile() -> int | None:
    try:
        return int(_pidfile_path().read_text().strip())
    except (OSError, ValueError):
        return None


def _write_pidfile() -> None:
    """Record this process as the canvas owner (atomic rename so a concurrent
    reader never sees a torn file). Cleared on exit via ``_clear_pidfile``."""
    path = _pidfile_path()
    try:
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(str(os.getpid()))
        tmp.replace(path)
        atexit.register(_clear_pidfile)
    except OSError:
        pass


def _clear_pidfile() -> None:
    """Remove the pidfile, but only if it still names us — so a successor that has
    already taken over (and rewritten it to its own pid) isn't clobbered."""
    path = _pidfile_path()
    try:
        if path.read_text().strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass


def _is_our_backend(pid: int) -> bool:
    """True only if ``pid`` is a live process whose cmdline is this project's
    backend. Read from ``/proc`` so we *never* terminate a process we can't
    positively identify as ours (a recycled pid, or an unrelated service)."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return "main.py" in raw.replace(b"\x00", b" ").decode("utf-8", "replace")


def _reclaim_canvas_port() -> bool:
    """Ensure THIS process can own the canvas port; return True if it can.

    If the port is free, done. If it's held by a *previous backend of ours*
    (per the pidfile + ``/proc`` check), terminate that instance and wait for the
    kernel to release the port. If it's held by anything we can't identify as
    ours, leave it alone and return False (caller runs headless / refuses)."""
    if _port_available(WS_HOST, WS_PORT):
        return True
    pid = _read_pidfile()
    if pid is None or pid == os.getpid() or not _is_our_backend(pid):
        return False
    log.warning("canvas port %d held by our previous backend (pid=%d); reclaiming it",
                WS_PORT, pid)
    # Poll the *port*, not the pid: a backend reaped by another parent can linger
    # as a zombie after its listening socket is already closed.
    for sig, grace in ((signal.SIGTERM, 4.0), (signal.SIGKILL, 2.0)):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except OSError:
            return False
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if _port_available(WS_HOST, WS_PORT):
                log.info("reclaimed canvas port %d from pid %d", WS_PORT, pid)
                return True
            time.sleep(0.05)
    log.warning("port %d still busy after terminating pid %d; running headless", WS_PORT, pid)
    return False


async def _serve_ws(server) -> None:
    """Run uvicorn, but never let a late bind failure (TOCTOU after the
    ``_port_available`` pre-check) take down the MCP stdio channel."""
    try:
        await server.serve()
    except Exception:  # noqa: BLE001 — bind error etc.; MCP must survive it
        log.exception("websocket server stopped unexpectedly; continuing MCP-only")


async def _run_full(store) -> None:
    """Full mode: MCP stdio loop + uvicorn, sharing one store + broadcast path.

    The MCP stdio loop is the primary: when its stdin reaches EOF (the MCP client
    — e.g. Claude Desktop — disconnected), ``run_stdio_async`` returns and we tell
    uvicorn to shut down so the whole process exits cleanly. On startup we
    ``_reclaim_canvas_port``: if a *previous backend of ours* still holds the
    port, we take it over so the human watches THIS session (the browser
    auto-reconnects to us). Only if the port is held by something we can't
    identify as ours do we run MCP-only and hand ``build_mcp`` a
    ``canvas_warning`` — so every mutation/``get_architecture`` reply still tells
    the agent (whose only channel is the MCP reply) the UI won't reflect its
    changes, instead of silently dropping them on the floor.
    """
    server, broadcast = _ws_server(store)
    canvas_ok = _reclaim_canvas_port()
    canvas_warning = None if canvas_ok else (
        f"live canvas unavailable: websocket port {WS_PORT} is held by another "
        f"process this backend can't identify as its own, so the UI will not "
        f"reflect these changes — stop that process or set WS_PORT")
    mcp = build_mcp(store, broadcast=broadcast, canvas_warning=canvas_warning)
    if canvas_ok:
        _write_pidfile()
        log.info("starting MCP stdio server + websocket on :%d", WS_PORT)
        server_task = asyncio.create_task(_serve_ws(server))
    else:
        log.warning("websocket port %d in use by a foreign process — running "
                    "MCP-only; tool responses now carry a 'live canvas "
                    "unavailable' warning so the agent knows the UI won't "
                    "update.", WS_PORT)
        server_task = None
    try:
        await mcp.run_stdio_async()
    finally:
        log.info("MCP stdio loop ended; shutting down websocket server")
        if server_task is not None:
            server.should_exit = True
            await server_task
        _clear_pidfile()


async def _run_standalone(store) -> None:
    """Standalone mode: websocket server only (frontend dev). Reclaims the canvas
    port from a previous backend of ours just like full mode; refuses to start a
    second canvas behind a foreign process holding the port."""
    server, _broadcast = _ws_server(store)
    if not _reclaim_canvas_port():
        log.error("canvas port %d is held by a process this backend can't "
                  "identify as its own; stop it or set WS_PORT. Refusing to "
                  "start a second canvas.", WS_PORT)
        raise SystemExit(1)
    _write_pidfile()
    log.info("starting websocket-only server on :%d (standalone)", WS_PORT)
    try:
        await server.serve()
    finally:
        _clear_pidfile()


def main() -> None:
    _configure_logging()
    mode = os.environ.get("MODE", "").lower()
    store = ArchitectureStore()

    if mode == "standalone":
        asyncio.run(_run_standalone(store))
    else:
        asyncio.run(_run_full(store))


if __name__ == "__main__":
    main()
