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
✅ done. `backend/mcp_tools.py` exposes `build_mcp(store, broadcast=None) -> FastMCP`, registering the
9 tools as thin adapters that each call one store method and return its dict. Mutating tools
(`add_layer`/`update_layer`/`remove_layer`/`connect_layers`/`disconnect_layers`/`reset_architecture`)
`await broadcast()` on success; reads (`get`/`validate`/`generate`) never broadcast. `broadcast`
defaults to a no-op (Stage 4 wires it to ws_app). `backend/main.py` builds ONE store and runs the
MCP stdio loop via `mcp.run_stdio_async()` under `asyncio.run`; `MODE=standalone` exits 1 with a
stderr notice (websocket server arrives Stage 4). All logging is forced onto stderr.

`backend/tests/test_mcp_tools.py` — 29 tests, all green
(`/home/hunain/base/bin/python -m pytest backend/tests/ -q` → 60 passed with Stage 2). They run
through a **real in-memory MCP client session** (`create_connected_server_and_client_session`):
exactly-9-tools + input-schema checks; happy path + every rejection branch for all 9 tools surfaced
as `isError` results; broadcast fires on successful mutations and not on reads/failed mutations;
valid-CNN / channel-mismatch / no-input validation; self-contained `generate_code`; reset restarts
ids; full build→validate→generate→reset round-trip.

Verification gate met:
- Real stdio transport (`mcp.client.stdio`) exercises all 9 tools end-to-end (build CNN → validate
  `[[2,10]]` → generate_code → error surfacing → reset). 
- Raw-subprocess purity check: every non-empty stdout line is a JSON-RPC 2.0 frame; our startup log
  lands on stderr and never on stdout.

Notes / deviations:
- Tool returns are annotated `dict[str, Any]` (not bare `dict`) so FastMCP emits **both**
  `structuredContent` and JSON text content; bare `dict` yields text only.
- Errors propagate as `ValueError` from the store; FastMCP wraps them as
  `Error executing tool <name>: <msg>` error results — no extra catch layer in the adapters.
- `validate_and_merge` intentionally does **not** type-check param *values* (only required-presence +
  unknown-key); value/shape correctness is real-execution's job (invariant 4), so the update_layer
  rejection test targets the unknown-param branch.
