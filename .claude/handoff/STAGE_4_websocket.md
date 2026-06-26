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
⬜ not started.
