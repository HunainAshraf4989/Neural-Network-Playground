# Stage 3 — MCP tools + main.py MCP loop

## Scope
Expose the store over MCP. `backend/mcp_tools.py` + the MCP half of `backend/main.py`.

## Files
- `backend/mcp_tools.py` — `FastMCP` (`from mcp.server.fastmcp import FastMCP`) registering the **9
  tools** via decorators, each a thin adapter calling one store method and returning its dict.
  Tool surface (see §9): `add_layer(type, params)`, `update_layer(node_id, params)`,
  `remove_layer(node_id)`, `connect_layers(from_id, to_id)`, `disconnect_layers(from_id, to_id)`,
  `get_architecture()`, `validate_architecture()`, `generate_code()`, `reset_architecture()`.
  After any successful mutation, trigger a websocket state broadcast (shared hook into ws_app, wired
  fully in Stage 4 — until then a no-op callback is fine).
- `backend/main.py` — construct ONE `ArchitectureStore`, pass it by reference to both adapters.
  Run modes via env var `MODE`: default = MCP stdio loop **+** uvicorn (Stage 4); `standalone` =
  uvicorn only. In Stage 3, default mode may run just the MCP stdio loop until ws_app exists.

## Invariants
**Nothing may print to stdout** except MCP framing — audit for stray prints; all logs → stderr.
Store is the only mutation path.

## Expected outcome
All 9 tools callable from an MCP client; build a CNN → validate → generate_code round-trips.

## Verification gate
- Launch MCP Inspector against `/home/hunain/base/bin/python backend/main.py` and exercise all 9
  tools (build a small CNN, validate true, generate code).
- Capture a run and assert stdout contains only protocol frames (no stray text). A quick check:
  pipe stderr/stdout separately and confirm logs land on stderr.

## Status
⬜ not started.
