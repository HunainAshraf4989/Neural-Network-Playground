# Stage 4 — WebSocket app + standalone server

## Scope
`backend/ws_app.py` (FastAPI app + ws handler) and the uvicorn half of `backend/main.py`. The ws
handler mirrors the MCP tool surface 1:1 over the **same store** — never separate mutation logic.

## Protocol (§12)
- **Server → client**, on connect and after every successful mutation, broadcast to all clients:
  `{"type": "state", "data": {"nodes": [...], "edges": [...]}}`.
- **Client → server** (human edits), one message per store method:
  `add_layer`/`update_layer`/`remove_layer`/`connect_layers`/`disconnect_layers`/`reset_architecture`
  (fields per §12).
- Server replies to the **originating client only** with `{"type": "ack", "ok": true}` or
  `{"type": "error", "message": "..."}`; and — only on success — broadcasts a fresh `state` to ALL
  clients including the originator.
- Wire the broadcast hook so **MCP mutations also trigger a state broadcast** (close the Stage 3 stub).

## main.py
Default mode: `asyncio.gather` the FastMCP stdio loop + `uvicorn.Server(...).serve()` on port
**8765**. `MODE=standalone`: uvicorn only (frontend dev without Claude Desktop).

## Expected outcome
`MODE=standalone` serves `ws://localhost:8765/ws`; a scripted client sees initial `state`, gets
`ack`, and receives a broadcast `state` after `add_layer`/`connect_layers`.

## Verification gate
```bash
MODE=standalone /home/hunain/base/bin/python backend/main.py &   # then:
/home/hunain/base/bin/python backend/tests/ws_smoke.py
```
`ws_smoke.py` connects, asserts initial state, sends a mutation, asserts ack + broadcast. Exit 0.

## Status
✅ done. `backend/ws_app.py` exposes `build_app(store) -> (app, broadcast)`: a FastAPI app with a
`/ws` endpoint and a `ConnectionManager` that tracks live clients. Each client message maps 1:1 to
one store method via the `_DISPATCH` table — no graph mutation lives in this module (invariant 2).
On connect the client gets the current `state`; on a successful mutation the originator gets
`{"type":"ack","ok":true}` and **all** clients (including the originator) get a fresh `state`
broadcast; rejections reply `{"type":"error","message":...}` to the originator only and never
broadcast. Malformed json, non-object messages, unknown `type`, and missing fields all return a clean
`error` without dropping the socket. The same `manager.broadcast` coroutine is handed to `build_mcp`,
so **MCP mutations broadcast to ws clients too** — the Stage 3 broadcast stub is now closed.

`backend/main.py`: default = full mode (`asyncio` MCP stdio loop + `uvicorn.Server` on :8765 sharing
one store + broadcast); `MODE=standalone` = uvicorn only. The MCP loop is primary: on stdin EOF
(client disconnect) it returns and we set `server.should_exit` so the whole process shuts down instead
of lingering. uvicorn gets `log_config=None` so its access log (which defaults to **stdout**) can't
corrupt the MCP frame stream — it propagates to the root logger (stderr + file) instead.

**Request logging (added per user request):** all logs go to stderr **and** a rotating file
(`LOG_FILE`, default `logs/nn_architect.log`); every MCP tool call and every ws message is logged at
INFO so a session can be replayed for debugging. `logs/` is gitignored.

Verification gate met: `MODE=standalone backend & ; backend/tests/ws_smoke.py` → exit 0 (initial
state, ack, broadcast all asserted over the real network transport).

Tests (all green; full suite 87 passed):
- `backend/tests/test_ws_app.py` — 25 in-process `TestClient` tests: initial state, ack+broadcast for
  all six mutations, an `error` (and no broadcast) for every rejection branch (unknown type, second
  input, unknown node, self-loop, unknown target, cycle, duplicate edge, second-input-into-non-merge,
  nonexistent edge), malformed-json / non-object / unknown-type / missing-field robustness, and
  multi-client broadcast fan-out + new-client-sees-existing-state.
- `backend/tests/test_ws_stdio.py` — 2 full-mode subprocess tests: (1) cross-surface single-store
  sync (MCP `add_layer` broadcasts to a live ws client; a ws mutation is visible to MCP
  `get_architecture`); (2) stdout stays pure JSON-RPC while uvicorn serves a ws client concurrently
  (uvicorn logs land on stderr, not stdout).
