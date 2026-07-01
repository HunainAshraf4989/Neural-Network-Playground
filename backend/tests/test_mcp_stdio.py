"""Real-stdio-transport guard for the MCP server (Stage 3 verification gate).

``test_mcp_tools.py`` exercises the tools through the *in-memory* session, which
never touches stdout. This module spawns ``backend/main.py`` as a genuine
subprocess and asserts the two properties only the real transport can prove:

1. **stdout purity (CLAUDE.md invariant 1).** Every non-empty stdout line (read
   while the session is live *and* drained to EOF afterwards) is a JSON-RPC 2.0
   frame. The highest-stakes invariant in the project, otherwise unguarded. The
   dangerous corruption is a *flushed* raw write to fd 1 (e.g.
   ``sys.stdout.write(...); flush()``) interleaving with frames — this test
   catches that. (A plain block-buffered ``print`` is largely swallowed by the
   SDK's separate stdout wrapper and rarely reaches the client, but the EOF drain
   still audits anything flushed at exit.) The check deliberately includes
   ``validate_architecture``, which spawns its *own* child subprocess, to prove
   that child's output can't leak onto our stdout either.
2. **Logs land on stderr, never stdout.**

A second test drives all 9 tools end-to-end over the real stdio client to prove
the full transport+adapter+store path round-trips (build CNN → validate →
generate_code → mutate → reset).
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


def asynctest(coro_fn):
    @functools.wraps(coro_fn)
    def wrapper():
        asyncio.run(coro_fn())
    return wrapper


def test_stdout_is_pure_jsonrpc_and_logs_go_to_stderr():
    """Drive the server over raw stdin/stdout and audit every stdout byte."""

    def line(obj):
        return json.dumps(obj) + "\n"

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "purity", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "add_layer",
                    "arguments": {"type": "input",
                                  "params": {"shape": [1, 8, 8], "dtype": "float32"}}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "add_layer", "arguments": {"type": "flatten", "params": {}}}},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
         "params": {"name": "connect_layers", "arguments": {"from_id": "n1", "to_id": "n2"}}},
        # spawns its own child subprocess internally — its output must not leak:
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
         "params": {"name": "validate_architecture", "arguments": {}}},
        # an error-producing call must still come back as a clean JSON-RPC frame:
        {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
         "params": {"name": "add_layer", "arguments": {"type": "bogus", "params": {}}}},
    ]
    expected_ids = {1, 2, 3, 4, 5, 6, 7}

    # WS_PORT=0 → a free ephemeral port, so this test never collides with a
    # running instance on the default port (which would flip the server into
    # MCP-only mode and change the startup log this test asserts on).
    env = {**os.environ, "WS_PORT": "0"}
    p = subprocess.Popen([PY, "main.py"], cwd=BACKEND, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    err_chunks = []
    threading.Thread(target=lambda: err_chunks.extend(p.stderr), daemon=True).start()

    try:
        for r in requests:
            p.stdin.write(line(r))
        p.stdin.flush()

        stdout_lines, ids_seen = [], set()
        # Read responses while stdin stays open; closing it early tears down
        # in-flight request handlers before they can answer.
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
        # Closing stdin lets the server exit cleanly; drain whatever remains so a
        # stray write that was only flushed at process exit is audited too.
        p.stdin.close()
        tail = p.stdout.read()
        stdout_lines.extend(tail.splitlines())
    finally:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()

    errlog = "".join(err_chunks)
    stdout_lines = [ln for ln in stdout_lines if ln.strip()]

    # every stdout line is a JSON-RPC 2.0 frame — no stray text whatsoever
    for ln in stdout_lines:
        obj = json.loads(ln)  # raises -> test fails if any line isn't JSON
        assert obj.get("jsonrpc") == "2.0", f"non-JSON-RPC line on stdout: {ln!r}"

    assert ids_seen >= expected_ids, f"missing responses; saw ids {sorted(ids_seen)}"
    # logs are on stderr, never stdout
    assert "starting MCP stdio server" in errlog, "startup log not found on stderr"
    assert "starting MCP stdio server" not in "\n".join(stdout_lines), "log leaked to stdout"

    # the validate call (id 6) actually ran (child subprocess) and reported back
    val = next(o for o in map(json.loads, stdout_lines) if o.get("id") == 6)
    assert val["result"]["structuredContent"]["valid"] is True
    # the bad call (id 7) surfaced as an MCP error result, not a transport break
    bad = next(o for o in map(json.loads, stdout_lines) if o.get("id") == 7)
    assert bad["result"]["isError"] is True


def test_websocket_bind_failure_does_not_kill_mcp():
    """Robustness gate: if the websocket port is already taken (a stale instance,
    a second session), the MCP stdio channel — the agent's only contract — must
    still come up and answer. We occupy a port, point the server at it via
    ``WS_PORT``, and assert ``tools/list`` still returns over stdio."""
    busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy.bind(("127.0.0.1", 0))
    busy.listen()
    busy_port = busy.getsockname()[1]

    env = os.environ.copy()
    env["WS_PORT"] = str(busy_port)

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "robust", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        # a real mutation must carry the "live canvas unavailable" warning back to
        # the agent (the agent never sees stderr — the MCP reply is its only channel)
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "add_layer",
                    "arguments": {"type": "input",
                                  "params": {"shape": [1, 8, 8], "dtype": "float32"}}}},
    ]

    p = subprocess.Popen([PY, "main.py"], cwd=BACKEND, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    out_lines, err_chunks = [], []
    threading.Thread(target=lambda: out_lines.extend(iter(p.stdout.readline, "")), daemon=True).start()
    threading.Thread(target=lambda: err_chunks.extend(p.stderr), daemon=True).start()

    def reply_for(req_id):
        for ln in list(out_lines):
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if obj.get("id") == req_id and "result" in obj:
                return obj
        return None

    try:
        for r in requests:
            p.stdin.write(json.dumps(r) + "\n")
        p.stdin.flush()

        # wait for the add_layer reply (id 3); it follows tools/list (id 2), so
        # seeing it means both arrived
        deadline, mutate = time.time() + 15, None
        while time.time() < deadline:
            mutate = reply_for(3)
            if mutate:
                break
            time.sleep(0.1)

        tools = reply_for(2)
        assert tools is not None, "MCP did not answer tools/list despite the websocket bind failing"
        assert len(tools["result"]["tools"]) == 12
        assert mutate is not None, "MCP did not answer add_layer despite the websocket bind failing"
        warnings = mutate["result"]["structuredContent"]["warnings"]
        assert any("canvas unavailable" in w.lower() for w in warnings), warnings
    finally:
        p.stdin.close()
        busy.close()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()

    errlog = "".join(err_chunks)
    # the busy port was detected and logged (not swallowed), and MCP kept serving
    assert "mcp-only" in errlog.lower() or "already in use" in errlog.lower()


def _wait_listening(port, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.25)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.1)
    return False


def test_new_backend_reclaims_canvas_port_from_stale_backend():
    """Single-canvas-owner guarantee: when a second backend starts on a port
    already held by a *previous nn-architect backend*, it reclaims the port
    (terminating the old instance) instead of running headless — so the human
    always watches the newest session and never has to hunt down stale processes.
    A *foreign* process is never killed (see the test above); this only fires for
    a process we can positively identify as our own backend."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    env = {**os.environ, "WS_PORT": str(port), "MODE": "standalone",
           "LOG_FILE": os.path.join(BACKEND, "_scratch", "test_reclaim.log")}

    def spawn():
        return subprocess.Popen([PY, "main.py"], cwd=BACKEND, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=env)

    procs = [spawn()]  # the "stale" backend
    try:
        assert _wait_listening(port), "first backend never came up"
        procs.append(spawn())  # the new backend should take the port over
        stale, fresh = procs

        deadline = time.time() + 15
        while time.time() < deadline and stale.poll() is None:
            time.sleep(0.1)
        assert stale.poll() is not None, "stale backend was not terminated by the new one"
        # the port is still served — by the new backend, which therefore reclaimed it
        assert _wait_listening(port), "new backend did not take over the canvas port"
        assert fresh.poll() is None, "new backend exited instead of serving"
    finally:
        for p in procs:
            p.stdin.close()
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


@asynctest
async def test_all_nine_tools_round_trip_over_real_stdio():
    """Build → validate → generate → mutate → reset through the real stdio client."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    def payload(res):
        assert res.isError is False, res.content[0].text
        return json.loads(res.content[0].text)

    params = StdioServerParameters(command=PY, args=["main.py"], cwd=BACKEND,
                                   env={**os.environ, "WS_PORT": "0"})  # free port: no collision
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            assert len((await session.list_tools()).tools) == 12

            async def add(t, prm=None):
                return payload(await session.call_tool(
                    "add_layer", {"type": t, "params": prm or {}}))["node_id"]

            async def conn(a, b):
                return payload(await session.call_tool(
                    "connect_layers", {"from_id": a, "to_id": b}))

            n1 = await add("input", {"shape": [1, 28, 28], "dtype": "float32"})
            n2 = await add("conv2d", {"in_channels": 1, "out_channels": 16, "kernel_size": 3})
            n3 = await add("maxpool2d", {"kernel_size": 2})
            n4 = await add("flatten")
            n5 = await add("linear", {"in_features": 16 * 13 * 13, "out_features": 10})
            for a, b in [(n1, n2), (n2, n3), (n3, n4), (n4, n5)]:
                await conn(a, b)

            up = payload(await session.call_tool(
                "update_layer", {"node_id": n2, "params": {"out_channels": 16}}))
            assert up["node_id"] == n2

            arch = payload(await session.call_tool("get_architecture", {}))
            assert [n["id"] for n in arch["nodes"]] == [n1, n2, n3, n4, n5]

            v = payload(await session.call_tool("validate_architecture", {}))
            assert v["valid"] is True and v["output_shapes"] == [[2, 10]]

            code = payload(await session.call_tool("generate_code", {}))["code"]
            assert "class LayerExecutionError" in code

            assert payload(await session.call_tool(
                "disconnect_layers", {"from_id": n4, "to_id": n5})) == {"removed": True}
            assert payload(await session.call_tool(
                "remove_layer", {"node_id": n5}))["removed"] is True

            assert payload(await session.call_tool("reset_architecture", {})) == {"reset": True}
            assert payload(await session.call_tool("get_architecture", {}))["nodes"] == []
            assert await add("input", {"shape": [1, 8, 8], "dtype": "float32"}) == "n1"
