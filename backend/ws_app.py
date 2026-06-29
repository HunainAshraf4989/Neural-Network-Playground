"""FastAPI websocket app — the human-facing half of the dual surface (spec §12).

The websocket handler is a **thin adapter over the same ``ArchitectureStore``** the
MCP tools call (CLAUDE.md invariant 2): every human edit maps 1:1 to one store
method, so the canvas can never drift from what the agent believes exists.

Protocol (§12):
  * **Server → client**: on connect, and after every successful mutation from
    *either* surface, a ``{"type": "state", "data": {nodes, edges}}`` message —
    broadcast to *all* connected clients.
  * **Client → server**: one message per mutating store method
    (``add_layer``/``update_layer``/``remove_layer``/``connect_layers``/
    ``disconnect_layers``/``reset_architecture``). The server replies to the
    *originating* client only with ``{"type":"ack","ok":true}`` or
    ``{"type":"error","message":...}``, and on success broadcasts a fresh state.

``build_app(store)`` returns ``(app, broadcast)``. The ``broadcast`` coroutine is
also handed to ``build_mcp`` so MCP mutations push state to ws clients too — one
broadcast path, shared. Every inbound request is logged (to stderr + the log file
configured in ``main.py``) so the whole session can be replayed for debugging;
stdout is never touched (invariant 1).
"""

import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

log = logging.getLogger("nn_architect.ws")


# Maps each client message ``type`` to (store method name, *field names* to pull
# from the message as positional args). The store is the only mutation path, so
# this table is the *entire* dispatch logic — no graph mutation lives here.
_DISPATCH = {
    "add_layer": ("add_layer", ("layer_type", "params")),
    "update_layer": ("update_layer", ("node_id", "params")),
    "remove_layer": ("remove_layer", ("node_id",)),
    "connect_layers": ("connect_layers", ("from_id", "to_id")),
    "disconnect_layers": ("disconnect_layers", ("from_id", "to_id")),
    "reset_architecture": ("reset_architecture", ()),
}


class ConnectionManager:
    """Tracks live websocket clients and pushes shared state to all of them.

    Lives entirely on the single asyncio event loop (the MCP stdio loop and the
    uvicorn server share it), so the client set needs no extra locking.
    """

    def __init__(self, store):
        self.store = store
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)
        log.info("client connected (%d total)", len(self.clients))
        await self._send_state(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)
        log.info("client disconnected (%d total)", len(self.clients))

    async def _state_message(self) -> dict:
        return {"type": "state", "data": await self.store.get_architecture()}

    async def _send_state(self, ws: WebSocket) -> None:
        await ws.send_json(await self._state_message())

    async def broadcast(self) -> None:
        """Push current state to every client. Used both by the ws handler (after
        a human edit) and by the MCP tools (after an agent edit) — the single
        broadcast path that keeps both surfaces in sync."""
        if not self.clients:
            return
        msg = await self._state_message()
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:  # client vanished mid-broadcast; prune it
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)
        log.info("broadcast state to %d client(s)", len(self.clients))


async def _apply(store, msg: dict):
    """Translate one client message into exactly one store call.

    Raises ``ValueError`` for an unknown type or a missing required field so the
    handler can surface it as a clean ``error`` reply (never a dropped socket).
    """
    msg_type = msg.get("type")
    entry = _DISPATCH.get(msg_type)
    if entry is None:
        raise ValueError(f"unknown message type '{msg_type}'")
    method_name, fields = entry
    try:
        args = [msg[f] for f in fields]
    except KeyError as missing:
        raise ValueError(f"'{msg_type}' is missing required field {missing}")
    return await getattr(store, method_name)(*args)


def build_app(store):
    """Build the FastAPI app and return ``(app, broadcast)``.

    ``broadcast`` is the manager's coroutine; ``main.py`` passes it to
    ``build_mcp`` so both surfaces share one broadcast path.
    """
    manager = ConnectionManager(store)
    app = FastAPI(title="nn-architect")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await manager.connect(ws)
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError as err:
                    log.warning("malformed json: %s", err)
                    await ws.send_json({"type": "error", "message": f"malformed json: {err}"})
                    continue
                if not isinstance(msg, dict):
                    await ws.send_json({"type": "error", "message": "message must be a json object"})
                    continue
                log.info("recv %s", msg)
                try:
                    await _apply(store, msg)
                except ValueError as err:
                    log.warning("rejected %s: %s", msg.get("type"), err)
                    await ws.send_json({"type": "error", "message": str(err)})
                    continue
                await ws.send_json({"type": "ack", "ok": True})
                await manager.broadcast()
        except WebSocketDisconnect:
            manager.disconnect(ws)
        except Exception:  # noqa: BLE001 — never let one client kill the server
            log.exception("ws handler error; closing client")
            manager.disconnect(ws)

    return app, manager.broadcast
