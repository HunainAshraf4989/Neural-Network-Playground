"""Stage 4 verification gate — scripted websocket client against a *running*
standalone backend (real network transport, not the in-process TestClient).

Usage (per the handoff gate):
    MODE=standalone /home/hunain/base/bin/python backend/main.py &
    /home/hunain/base/bin/python backend/tests/ws_smoke.py

Connects to ``ws://localhost:8765/ws``, asserts the initial ``state``, sends an
``add_layer`` then a ``connect_layers``, and for each asserts the ``ack`` plus the
broadcast ``state`` reflecting the change. Exits 0 on success, 1 on any failure.
A short connect-retry loop tolerates the server still booting.
"""

import asyncio
import json
import sys

import websockets

URL = "ws://localhost:8765/ws"


async def _recv(ws):
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=5))


async def main() -> int:
    # the server may still be coming up when this script starts
    last_err = None
    ws = None
    for _ in range(50):
        try:
            ws = await websockets.connect(URL)
            break
        except (OSError, websockets.WebSocketException) as err:  # noqa: PERF203
            last_err = err
            await asyncio.sleep(0.1)
    if ws is None:
        print(f"FAIL: could not connect to {URL}: {last_err}", file=sys.stderr)
        return 1

    async with ws:
        # 1. initial state on connect — empty graph
        state = await _recv(ws)
        assert state["type"] == "state", state
        assert state["data"] == {"nodes": [], "edges": []}, state["data"]

        # 2. add_layer (input) -> ack, then broadcast state with the node
        await ws.send(json.dumps({"type": "add_layer", "layer_type": "input",
                                  "params": {"shape": [1, 8, 8], "dtype": "float32"}}))
        assert await _recv(ws) == {"type": "ack", "ok": True}
        data = (await _recv(ws))["data"]
        assert [n["id"] for n in data["nodes"]] == ["n1"], data

        # 3. add a second node + connect -> ack, then broadcast with the edge
        await ws.send(json.dumps({"type": "add_layer", "layer_type": "flatten",
                                  "params": {}}))
        assert await _recv(ws) == {"type": "ack", "ok": True}
        await _recv(ws)  # broadcast for the add
        await ws.send(json.dumps({"type": "connect_layers", "from_id": "n1",
                                  "to_id": "n2"}))
        assert await _recv(ws) == {"type": "ack", "ok": True}
        data = (await _recv(ws))["data"]
        assert data["edges"] == [{"from": "n1", "to": "n2"}], data

    print("ws_smoke OK: initial state, ack, and broadcast all verified")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except AssertionError as err:
        print(f"FAIL: assertion failed: {err!r}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:  # noqa: BLE001
        print(f"FAIL: {err!r}", file=sys.stderr)
        sys.exit(1)
