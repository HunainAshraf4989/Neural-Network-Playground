"""Process entry point - constructs the single ``ArchitectureStore`` and runs the
chosen server mode (spec §11; CLAUDE.md invariant 2: one store, one mutation path).

``MODE`` env var:
  (default)   full mode: MCP stdio loop **and** the uvicorn/websocket server on
              :8765, sharing one store and one broadcast path. This is what
              Claude Desktop launches.
  standalone  websocket-only server (frontend dev without Claude Desktop).

**Busy port: warn, never fight.** There must be exactly one live canvas, and the
canvas port is how you get it. This backend never tries to take the port from
whoever already holds it - it does one cross-platform bind probe
(``_port_available``, plain sockets, so Linux/macOS/Windows behave the same) and,
if the port is busy, says so loudly instead of starting a second, invisible
canvas. Standalone mode refuses to start. Full mode keeps serving MCP (the
agent's only contract) but runs headless, and threads a ``canvas_warning`` into
``build_mcp`` so *every* mutation and ``get_architecture`` reply tells the agent
the UI won't reflect its changes - the MCP reply is the only channel that reaches
a human here, since Claude Desktop launches this process with no terminal
attached. Either way the remedy is the same and is named in the message: free the
port, or change ``DEFAULT_WS_PORT`` in ``config.py``.

Trade-off worth knowing: when the squatter is a *stale nn-architect backend* (a
Claude Desktop respawn, say), the browser stays attached to that older process,
so the human watches a canvas the current agent isn't writing to. The warning is
what tells you to go clear it.

stdout is reserved for MCP protocol framing (invariant 1). All logging is forced
onto stderr **and** a rotating log file (``LOG_FILE``) so no library can leak a
line onto stdout, and so every request can be replayed for debugging. uvicorn is
given ``log_config=None`` for the same reason: its default config writes the
access log to *stdout*, which would corrupt the MCP stream in full mode - with no
config it propagates to the root logger (stderr + file) instead.
"""

import asyncio
import logging
import logging.handlers
import socket
import sys
from pathlib import Path

import uvicorn

from config import load as load_config
from mcp_tools import build_mcp
from store import ArchitectureStore
from ws_app import build_app

log = logging.getLogger("nn_architect.main")

CONFIG = load_config()
WS_HOST = "0.0.0.0"
WS_PORT = CONFIG.ws_port


def _configure_logging() -> None:
    """Send all logs to stderr and to a rotating file; never to stdout."""
    level = CONFIG.log_level
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)
    handlers = [stderr_handler]

    log_file = CONFIG.log_file
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


# -- the one message a user sees when the port is taken ------------------------

def _busy_port_message() -> str:
    """The remedy, phrased once, for whoever can see it: the agent (via the MCP
    ``canvas_warning``) in full mode, the terminal in standalone mode."""
    return (f"port {WS_PORT} is already in use, so the live canvas is unavailable. "
            f"Free that port - stop whatever is listening on it, which is usually an "
            f"older nn-architect backend - or pick a different one by changing "
            f"DEFAULT_WS_PORT in backend/config.py (the frontend reads it from there "
            f"too), then restart.")


async def _serve_ws(server) -> None:
    """Run uvicorn, but never let a late bind failure (TOCTOU after the
    ``_port_available`` pre-check) take down the MCP stdio channel."""
    try:
        await server.serve()
    except Exception:  # noqa: BLE001 - bind error etc.; MCP must survive it
        log.exception("websocket server stopped unexpectedly; continuing MCP-only")


async def _run_full(store) -> None:
    """Full mode: MCP stdio loop + uvicorn, sharing one store + broadcast path.

    The MCP stdio loop is the primary: when its stdin reaches EOF (the MCP client
    - e.g. Claude Desktop - disconnected), ``run_stdio_async`` returns and we tell
    uvicorn to shut down so the whole process exits cleanly. If the canvas port is
    already taken we do *not* try to wrestle it away: we run MCP-only and hand
    ``build_mcp`` a ``canvas_warning``, so every mutation/``get_architecture``
    reply tells the agent (whose only channel is the MCP reply) that the UI won't
    reflect its changes and how to fix it, instead of silently dropping the
    updates on the floor.
    """
    server, broadcast = _ws_server(store)
    canvas_ok = _port_available(WS_HOST, WS_PORT)
    canvas_warning = None if canvas_ok else _busy_port_message()
    mcp = build_mcp(store, broadcast=broadcast, canvas_warning=canvas_warning)
    if canvas_ok:
        log.info("starting MCP stdio server + websocket on :%d", WS_PORT)
        server_task = asyncio.create_task(_serve_ws(server))
    else:
        log.warning("running MCP-only: %s Tool responses now carry this warning "
                    "so the agent knows the UI won't update.", _busy_port_message())
        server_task = None
    try:
        await mcp.run_stdio_async()
    finally:
        log.info("MCP stdio loop ended; shutting down websocket server")
        if server_task is not None:
            server.should_exit = True
            await server_task


async def _run_standalone(store) -> None:
    """Standalone mode: websocket server only (frontend dev). Refuses to start
    behind a busy port - there's a terminal here, so a loud exit is the clearest
    signal, and starting a second invisible canvas would help nobody."""
    server, _broadcast = _ws_server(store)
    if not _port_available(WS_HOST, WS_PORT):
        log.error("refusing to start a second canvas: %s", _busy_port_message())
        raise SystemExit(1)
    log.info("starting websocket-only server on :%d (standalone)", WS_PORT)
    await server.serve()


def main() -> None:
    _configure_logging()
    mode = CONFIG.mode
    store = ArchitectureStore()

    if mode == "standalone":
        asyncio.run(_run_standalone(store))
    else:
        asyncio.run(_run_full(store))


if __name__ == "__main__":
    main()
