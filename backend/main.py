"""Process entry point — constructs the single ``ArchitectureStore`` and runs the
chosen server mode (spec §11; CLAUDE.md invariant 2: one store, one mutation path).

``MODE`` env var:
  (default)   MCP stdio loop. Stage 4 adds the uvicorn/websocket server alongside.
  standalone  websocket-only server (frontend dev). Arrives in Stage 4.

stdout is reserved for MCP protocol framing (invariant 1). All logging is forced
onto stderr so no library can leak a line onto stdout and corrupt the stream.
"""

import asyncio
import logging
import os
import sys

from mcp_tools import build_mcp
from store import ArchitectureStore

log = logging.getLogger("nn_architect.main")


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


async def _run_mcp(store: ArchitectureStore) -> None:
    # broadcast hook stays a no-op until Stage 4 wires it to the websocket app.
    mcp = build_mcp(store)
    await mcp.run_stdio_async()


def main() -> None:
    _configure_logging()
    mode = os.environ.get("MODE", "").lower()
    store = ArchitectureStore()

    if mode == "standalone":
        log.error("MODE=standalone (websocket-only) is implemented in Stage 4; "
                  "nothing to serve yet")
        sys.exit(1)

    log.info("starting MCP stdio server")
    asyncio.run(_run_mcp(store))


if __name__ == "__main__":
    main()
