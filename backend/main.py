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
import sys
from pathlib import Path

import uvicorn

from mcp_tools import build_mcp
from store import ArchitectureStore
from ws_app import build_app

log = logging.getLogger("nn_architect.main")

_REPO_ROOT = Path(__file__).resolve().parent.parent
WS_HOST = "0.0.0.0"
WS_PORT = 8765


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


async def _run_full(store) -> None:
    """Full mode: MCP stdio loop + uvicorn, sharing one store + broadcast path.

    The MCP stdio loop is the primary: when its stdin reaches EOF (the MCP client
    — e.g. Claude Desktop — disconnected), ``run_stdio_async`` returns and we tell
    uvicorn to shut down so the whole process exits cleanly instead of lingering
    on a half-dead connection.
    """
    server, broadcast = _ws_server(store)
    mcp = build_mcp(store, broadcast=broadcast)
    log.info("starting MCP stdio server + websocket on :%d", WS_PORT)
    server_task = asyncio.create_task(server.serve())
    try:
        await mcp.run_stdio_async()
    finally:
        log.info("MCP stdio loop ended; shutting down websocket server")
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
