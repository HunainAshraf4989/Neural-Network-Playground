"""Full-mode (MCP stdio + uvicorn) integration guards for Stage 4.

``test_ws_app.py`` proves the ws adapter in isolation; this module spawns
``backend/main.py`` in **full mode** (the real dual surface Claude Desktop runs)
and proves the two things only the combined process can show:

1. **Cross-surface single-store sync (CLAUDE.md invariant 2).** An ``add_layer``
   issued over the *MCP stdio* transport broadcasts a fresh ``state`` to a live
   *websocket* client; and a mutation sent over the *websocket* is then visible to
   an MCP ``get_architecture``. Same store, both directions, one broadcast path.

2. **stdout purity with uvicorn co-resident (invariant 1).** uvicorn's default log
   config writes its access log to *stdout*, which would corrupt the MCP frame
   stream. ``main.py`` passes ``log_config=None`` to stop that; this test drives
   MCP over raw stdin/stdout while a ws client generates uvicorn activity, then
   asserts every stdout line is a JSON-RPC 2.0 frame and uvicorn's logs landed on
   stderr instead.

Each test binds its own free ephemeral port (so it never collides with a server
already running on the default port) and fully tears the server down before the
next starts.
"""

import asyncio
import functools
import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

PY = sys.executable
BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _free_port():
    """Reserve a free ephemeral port, then release it for main.py to bind. The
    tiny TOCTOU window is fine for a test, and a dedicated port keeps these tests
    from colliding with any server already running on the default port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _env(port):
    """Full mode (no MODE) on a dedicated port, request log to a throwaway file."""
    env = os.environ.copy()
    env.pop("MODE", None)
    env["WS_PORT"] = str(port)
    env["LOG_FILE"] = os.path.join(BACKEND, "_scratch", "test_ws_stdio.log")
    return env


def _wait_for_port(port, host="localhost", timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.25)
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.1)
    return False


def asynctest(coro_fn):
    @functools.wraps(coro_fn)
    def wrapper():
        asyncio.run(coro_fn())
    return wrapper


@asynctest
async def test_cross_surface_sync_mcp_and_ws_share_one_store():
    """MCP edit -> ws broadcast; ws edit -> visible to MCP get_architecture."""
    import websockets
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    def payload(res):
        assert res.isError is False, res.content[0].text
        return json.loads(res.content[0].text)

    async def recv(ws):
        return json.loads(await asyncio.wait_for(ws.recv(), timeout=10))

    port = _free_port()
    ws_url = f"ws://localhost:{port}/ws"
    params = StdioServerParameters(command=PY, args=["main.py"], cwd=BACKEND, env=_env(port))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            assert _wait_for_port(port), "websocket server never came up in full mode"

            async with websockets.connect(ws_url) as ws:
                # initial state on connect - empty
                assert (await recv(ws))["data"] == {"nodes": [], "edges": [], "layout": {}}

                # (1) MCP mutation must broadcast to the ws client
                payload(await session.call_tool(
                    "add_layer",
                    {"type": "input", "params": {"shape": [1, 8, 8], "dtype": "float32"}}))
                pushed = await recv(ws)
                assert pushed["type"] == "state"
                assert [n["id"] for n in pushed["data"]["nodes"]] == ["n1"]

                # (2) ws mutation must be visible to MCP (same store)
                await ws.send(json.dumps({"type": "add_layer",
                                          "layer_type": "flatten", "params": {}}))
                assert await recv(ws) == {"type": "ack", "ok": True}
                await recv(ws)  # broadcast for the ws add

                arch = payload(await session.call_tool("get_architecture", {}))
                assert [n["id"] for n in arch["nodes"]] == ["n1", "n2"]


def test_full_mode_stdout_is_pure_jsonrpc_despite_uvicorn():
    """Drive MCP over raw stdout while uvicorn serves a ws client; audit stdout."""
    from websockets.sync.client import connect as ws_connect

    def line(obj):
        return json.dumps(obj) + "\n"

    # Build a deterministic, known-valid graph entirely over MCP (input -> flatten)
    # so the validate assertion never depends on the ws-added node below.
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "purity", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "add_layer",
                    "arguments": {"type": "input",
                                  "params": {"shape": [1, 8, 8], "dtype": "float32"}}}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "add_layer", "arguments": {"type": "flatten", "params": {}}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "connect_layers",
                    "arguments": {"from_id": "n1", "to_id": "n2"}}},
        # validate spawns its own child subprocess - its output must not leak either
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
         "params": {"name": "validate_architecture", "arguments": {}}},
    ]
    expected_ids = {1, 2, 3, 4, 5}

    port = _free_port()
    ws_url = f"ws://localhost:{port}/ws"
    p = subprocess.Popen([PY, "main.py"], cwd=BACKEND, env=_env(port),
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    err_chunks = []
    threading.Thread(target=lambda: err_chunks.extend(p.stderr), daemon=True).start()

    try:
        for r in requests:
            p.stdin.write(line(r))
        p.stdin.flush()

        # generate uvicorn activity: a real ws client connecting + mutating
        # exercises the access path that defaults to stdout. Any leak it produces
        # is caught in the stdout drain below.
        assert _wait_for_port(port), "websocket server never came up in full mode"
        with ws_connect(ws_url) as ws:
            json.loads(ws.recv())  # initial state (already has n1, n2 from MCP)
            ws.send(json.dumps({"type": "add_layer", "layer_type": "dropout",
                                "params": {}}))
            assert json.loads(ws.recv()) == {"type": "ack", "ok": True}
            json.loads(ws.recv())  # broadcast

        stdout_lines, ids_seen = [], set()
        while not (ids_seen >= expected_ids):
            ln = p.stdout.readline()
            if not ln:
                break
            stdout_lines.append(ln.rstrip("\n"))
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "id" in obj:
                ids_seen.add(obj["id"])
        p.stdin.close()
        stdout_lines.extend(p.stdout.read().splitlines())
    finally:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()

    errlog = "".join(err_chunks)
    stdout_lines = [ln for ln in stdout_lines if ln.strip()]

    # every stdout line is a JSON-RPC frame - uvicorn's access log did NOT leak
    for ln in stdout_lines:
        obj = json.loads(ln)  # raises -> fails if any line isn't JSON
        assert obj.get("jsonrpc") == "2.0", f"non-JSON-RPC line on stdout: {ln!r}"

    assert ids_seen >= expected_ids, f"missing responses; saw {sorted(ids_seen)}"
    # uvicorn's own logs landed on stderr, never stdout
    assert "Uvicorn running" not in "\n".join(stdout_lines), "uvicorn leaked to stdout"
    assert "uvicorn" in errlog.lower(), "expected uvicorn logs on stderr"
    # validate (id 5) actually ran its child subprocess and reported back cleanly
    val = next(o for o in map(json.loads, stdout_lines) if o.get("id") == 5)
    assert val["result"]["structuredContent"]["valid"] is True
