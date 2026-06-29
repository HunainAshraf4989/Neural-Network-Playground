"""Process entry point — constructs the single ``ArchitectureStore`` and runs the
chosen server mode (spec §11; CLAUDE.md invariant 2: one store, one mutation path).

``MODE`` env var:
  (default)   full mode: MCP stdio loop **and** the uvicorn/websocket server on
              :8765, sharing one store and one broadcast path. This is what
              Claude Desktop launches.
  standalone  websocket-only server (frontend dev without Claude Desktop).

stdout is reserved for MCP protocol framing (invariant 1). All logging is forced
onto stderr **and** a rotating log file (``LOG_FILE``) so no library can leak a
line onto stdout, and so every request can be replayed for debugging. uvicorn is
given ``log_config=None`` for the same reason: its default config writes the
access log to *stdout*, which would corrupt the MCP stream in full mode — with no
config it propagates to the root logger (stderr + file) instead.
"""

import asyncio
import logging
import logging.handlers
import os
import socket
import sys
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
    uvicorn to shut down so the whole process exits cleanly. If the websocket port
    is already in use (a stale or duplicate instance), we DON'T start uvicorn at
    all — that keeps a busy port from ever silently killing the MCP connection.
    Instead we hand ``build_mcp`` a ``canvas_warning`` so every mutation/
    ``get_architecture`` reply tells the agent the UI won't reflect its changes
    (a busy port means a *rival* store owns the canvas — the agent's edits would
    otherwise vanish silently, which is exactly the failure this surfaces).
    """
    server, broadcast = _ws_server(store)
    canvas_ok = _port_available(WS_HOST, WS_PORT)
    canvas_warning = None if canvas_ok else (
        f"live canvas unavailable: websocket port {WS_PORT} is already in use "
        f"(another nn-architect backend owns it), so the UI will not reflect "
        f"these changes — stop the other backend or set WS_PORT")
    mcp = build_mcp(store, broadcast=broadcast, canvas_warning=canvas_warning)
    if canvas_ok:
        log.info("starting MCP stdio server + websocket on :%d", WS_PORT)
        server_task = asyncio.create_task(_serve_ws(server))
    else:
        log.warning("websocket port %d already in use — running MCP-only; tool "
                    "responses now carry a 'live canvas unavailable' warning so "
                    "the agent knows the UI won't update.", WS_PORT)
        server_task = None
    try:
        await mcp.run_stdio_async()
    finally:
        log.info("MCP stdio loop ended; shutting down websocket server")
        if server_task is not None:
            server.should_exit = True
            await server_task


async def _run_standalone(store) -> None:
    """Standalone mode: websocket server only (frontend dev)."""
    server, _broadcast = _ws_server(store)
    log.info("starting websocket-only server on :%d (standalone)", WS_PORT)
    await server.serve()


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
