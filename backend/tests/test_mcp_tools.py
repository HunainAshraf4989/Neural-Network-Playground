"""End-to-end tests for the MCP tool surface (Stage 3).

These exercise the tools the way Claude Desktop does: through a real in-memory
MCP client session connected to the FastMCP server from ``build_mcp``. That means
we cover the full path — argument schema validation, the thin adapter, the store,
and the way errors surface as MCP error results (``isError: true``) — not just
direct Python calls.

The store's structural rules themselves are unit-tested exhaustively in
``test_store.py``; here we assert that **every tool** is wired correctly, returns
its §9 shape, surfaces each rejection branch as an MCP error, and that the
broadcast hook fires after successful mutations only.

Async tests are wrapped in ``asyncio.run`` via ``@asynctest`` (no pytest-asyncio
dependency). Everything per test is built inside the coroutine so the store's
Lock and the session bind to that run's event loop.
"""

import asyncio
import contextlib
import functools
import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session as connect

from mcp_tools import build_mcp
from store import ArchitectureStore


def asynctest(coro_fn):
    @functools.wraps(coro_fn)
    def wrapper():
        asyncio.run(coro_fn())
    return wrapper


class _Counter:
    """Async, no-arg broadcast hook that counts how often it fired."""
    def __init__(self):
        self.n = 0

    async def __call__(self):
        self.n += 1


@contextlib.asynccontextmanager
async def harness():
    """Yield ``(session, store, broadcast_counter)`` over a live in-memory MCP
    session backed by a fresh store."""
    store = ArchitectureStore()
    counter = _Counter()
    mcp = build_mcp(store, broadcast=counter)
    async with connect(mcp) as session:
        yield session, store, counter


def ok(result):
    """Assert the tool call succeeded and return its decoded payload dict."""
    assert result.isError is False, getattr(result.content[0], "text", result.content)
    return json.loads(result.content[0].text)


def err(result):
    """Assert the tool call returned an MCP error and return the message text."""
    assert result.isError is True, f"expected error, got: {result.content}"
    return result.content[0].text


async def _add(session, type, params=None):
    return ok(await session.call_tool("add_layer", {"type": type, "params": params or {}}))["node_id"]


async def _build_cnn(session):
    """input(1,28,28) -> conv2d -> maxpool2d -> flatten -> linear(10)."""
    n1 = await _add(session, "input", {"shape": [1, 28, 28], "dtype": "float32"})
    n2 = await _add(session, "conv2d", {"in_channels": 1, "out_channels": 16, "kernel_size": 3})
    n3 = await _add(session, "maxpool2d", {"kernel_size": 2})
    n4 = await _add(session, "flatten")
    n5 = await _add(session, "linear", {"in_features": 16 * 13 * 13, "out_features": 10})
    for a, b in [(n1, n2), (n2, n3), (n3, n4), (n4, n5)]:
        ok(await session.call_tool("connect_layers", {"from_id": a, "to_id": b}))
    return [n1, n2, n3, n4, n5]


# --------------------------------------------------------------------------- #
# tool registration / schema
# --------------------------------------------------------------------------- #
EXPECTED_TOOLS = {
    "add_layer", "update_layer", "remove_layer", "connect_layers",
    "disconnect_layers", "get_architecture", "validate_architecture",
    "generate_code", "reset_architecture",
}


@asynctest
async def test_exactly_nine_tools_registered():
    async with harness() as (session, _store, _c):
        tools = (await session.list_tools()).tools
        assert {t.name for t in tools} == EXPECTED_TOOLS
        assert len(tools) == 9


@asynctest
async def test_tool_input_schemas():
    async with harness() as (session, _store, _c):
        by_name = {t.name: t for t in (await session.list_tools()).tools}
        assert set(by_name["add_layer"].inputSchema["required"]) == {"type", "params"}
        assert set(by_name["update_layer"].inputSchema["required"]) == {"node_id", "params"}
        assert set(by_name["connect_layers"].inputSchema["required"]) == {"from_id", "to_id"}
        assert set(by_name["remove_layer"].inputSchema["required"]) == {"node_id"}
        # reads take no params
        for name in ("get_architecture", "validate_architecture", "generate_code",
                     "reset_architecture"):
            assert not by_name[name].inputSchema.get("required")
        # every tool carries a description (agent-facing docs)
        assert all(t.description for t in by_name.values())


# --------------------------------------------------------------------------- #
# add_layer
# --------------------------------------------------------------------------- #
@asynctest
async def test_add_layer_happy_and_broadcast():
    async with harness() as (session, _store, c):
        res = ok(await session.call_tool("add_layer",
                                         {"type": "input",
                                          "params": {"shape": [1, 28, 28], "dtype": "float32"}}))
        assert res == {"node_id": "n1"}
        assert c.n == 1


@asynctest
async def test_add_layer_second_input_rejected_no_broadcast():
    async with harness() as (session, _store, c):
        await _add(session, "input", {"shape": [1, 28, 28], "dtype": "float32"})
        assert c.n == 1
        msg = err(await session.call_tool("add_layer",
                                          {"type": "input",
                                           "params": {"shape": [3, 8, 8], "dtype": "float32"}}))
        assert "input node already exists" in msg
        assert c.n == 1  # failed mutation does not broadcast


@asynctest
async def test_add_layer_unknown_type_rejected():
    async with harness() as (session, _store, _c):
        msg = err(await session.call_tool("add_layer", {"type": "not_a_layer", "params": {}}))
        assert "unknown layer type" in msg


@asynctest
async def test_add_layer_missing_required_param_rejected():
    async with harness() as (session, _store, _c):
        msg = err(await session.call_tool("add_layer", {"type": "conv2d", "params": {}}))
        assert "missing required param" in msg


# --------------------------------------------------------------------------- #
# update_layer
# --------------------------------------------------------------------------- #
@asynctest
async def test_update_layer_merges_and_broadcasts():
    async with harness() as (session, _store, c):
        nid = await _add(session, "conv2d",
                         {"in_channels": 1, "out_channels": 8, "kernel_size": 3})
        before = c.n
        res = ok(await session.call_tool("update_layer",
                                         {"node_id": nid, "params": {"out_channels": 32}}))
        assert res["node_id"] == nid
        assert res["params"]["out_channels"] == 32
        assert res["params"]["in_channels"] == 1  # untouched fields preserved
        assert c.n == before + 1


@asynctest
async def test_update_layer_unknown_node_rejected():
    async with harness() as (session, _store, _c):
        msg = err(await session.call_tool("update_layer",
                                          {"node_id": "n99", "params": {"out_channels": 4}}))
        assert "unknown node" in msg


@asynctest
async def test_update_layer_unknown_param_rejected():
    async with harness() as (session, _store, _c):
        nid = await _add(session, "conv2d",
                         {"in_channels": 1, "out_channels": 8, "kernel_size": 3})
        msg = err(await session.call_tool("update_layer",
                                          {"node_id": nid, "params": {"bogus": 1}}))
        assert "unknown param" in msg


# --------------------------------------------------------------------------- #
# remove_layer
# --------------------------------------------------------------------------- #
@asynctest
async def test_remove_layer_cascades_edges_and_broadcasts():
    async with harness() as (session, _store, c):
        n1 = await _add(session, "input", {"shape": [1, 28, 28], "dtype": "float32"})
        n2 = await _add(session, "relu")
        ok(await session.call_tool("connect_layers", {"from_id": n1, "to_id": n2}))
        before = c.n
        res = ok(await session.call_tool("remove_layer", {"node_id": n2}))
        assert res["removed"] is True
        assert {"from": n1, "to": n2} in res["edges_removed"]
        assert c.n == before + 1
        arch = ok(await session.call_tool("get_architecture", {}))
        assert arch["edges"] == []
        assert [n["id"] for n in arch["nodes"]] == [n1]


@asynctest
async def test_remove_layer_unknown_node_rejected():
    async with harness() as (session, _store, _c):
        msg = err(await session.call_tool("remove_layer", {"node_id": "n42"}))
        assert "unknown node" in msg


# --------------------------------------------------------------------------- #
# connect_layers
# --------------------------------------------------------------------------- #
@asynctest
async def test_connect_layers_happy_and_broadcast():
    async with harness() as (session, _store, c):
        n1 = await _add(session, "input", {"shape": [1, 28, 28], "dtype": "float32"})
        n2 = await _add(session, "relu")
        before = c.n
        res = ok(await session.call_tool("connect_layers", {"from_id": n1, "to_id": n2}))
        assert res["edge"] == {"from": n1, "to": n2}
        assert c.n == before + 1


@asynctest
async def test_connect_layers_unknown_source_and_target():
    async with harness() as (session, _store, _c):
        n1 = await _add(session, "input", {"shape": [1, 28, 28], "dtype": "float32"})
        assert "unknown source" in err(
            await session.call_tool("connect_layers", {"from_id": "nX", "to_id": n1}))
        assert "unknown target" in err(
            await session.call_tool("connect_layers", {"from_id": n1, "to_id": "nX"}))


@asynctest
async def test_connect_layers_self_loop_rejected():
    async with harness() as (session, _store, _c):
        n2 = await _add(session, "relu")
        assert "self-loop" in err(
            await session.call_tool("connect_layers", {"from_id": n2, "to_id": n2}))


@asynctest
async def test_connect_layers_into_input_rejected():
    async with harness() as (session, _store, _c):
        n1 = await _add(session, "input", {"shape": [1, 28, 28], "dtype": "float32"})
        n2 = await _add(session, "relu")
        assert "input node" in err(
            await session.call_tool("connect_layers", {"from_id": n2, "to_id": n1}))


@asynctest
async def test_connect_layers_duplicate_rejected_no_broadcast():
    async with harness() as (session, _store, c):
        n1 = await _add(session, "input", {"shape": [1, 28, 28], "dtype": "float32"})
        n2 = await _add(session, "relu")
        ok(await session.call_tool("connect_layers", {"from_id": n1, "to_id": n2}))
        before = c.n
        assert "already exists" in err(
            await session.call_tool("connect_layers", {"from_id": n1, "to_id": n2}))
        assert c.n == before


@asynctest
async def test_connect_layers_second_inbound_into_non_merge_rejected():
    async with harness() as (session, _store, _c):
        n1 = await _add(session, "input", {"shape": [8, 8, 8], "dtype": "float32"})
        n2 = await _add(session, "relu")
        n3 = await _add(session, "relu")
        ok(await session.call_tool("connect_layers", {"from_id": n1, "to_id": n3}))
        assert "exactly one input" in err(
            await session.call_tool("connect_layers", {"from_id": n2, "to_id": n3}))


@asynctest
async def test_connect_layers_merge_node_accepts_multiple_inbound():
    async with harness() as (session, _store, _c):
        n1 = await _add(session, "input", {"shape": [8, 8, 8], "dtype": "float32"})
        n2 = await _add(session, "relu")
        m = await _add(session, "add")
        ok(await session.call_tool("connect_layers", {"from_id": n1, "to_id": m}))
        ok(await session.call_tool("connect_layers", {"from_id": n2, "to_id": m}))
        arch = ok(await session.call_tool("get_architecture", {}))
        assert sum(1 for e in arch["edges"] if e["to"] == m) == 2


@asynctest
async def test_connect_layers_cycle_rejected():
    async with harness() as (session, _store, _c):
        a = await _add(session, "relu")
        b = await _add(session, "relu")
        ok(await session.call_tool("connect_layers", {"from_id": a, "to_id": b}))
        assert "cycle" in err(
            await session.call_tool("connect_layers", {"from_id": b, "to_id": a}))


# --------------------------------------------------------------------------- #
# disconnect_layers
# --------------------------------------------------------------------------- #
@asynctest
async def test_disconnect_layers_happy_and_broadcast():
    async with harness() as (session, _store, c):
        n1 = await _add(session, "input", {"shape": [1, 28, 28], "dtype": "float32"})
        n2 = await _add(session, "relu")
        ok(await session.call_tool("connect_layers", {"from_id": n1, "to_id": n2}))
        before = c.n
        res = ok(await session.call_tool("disconnect_layers", {"from_id": n1, "to_id": n2}))
        assert res == {"removed": True}
        assert c.n == before + 1
        assert ok(await session.call_tool("get_architecture", {}))["edges"] == []


@asynctest
async def test_disconnect_layers_absent_edge_rejected():
    async with harness() as (session, _store, _c):
        msg = err(await session.call_tool("disconnect_layers", {"from_id": "n1", "to_id": "n2"}))
        assert "does not exist" in msg


# --------------------------------------------------------------------------- #
# get_architecture (read, no broadcast)
# --------------------------------------------------------------------------- #
@asynctest
async def test_get_architecture_shape_and_no_broadcast():
    async with harness() as (session, _store, c):
        await _build_cnn(session)
        before = c.n
        arch = ok(await session.call_tool("get_architecture", {}))
        assert [n["id"] for n in arch["nodes"]] == ["n1", "n2", "n3", "n4", "n5"]
        assert arch["edges"][0] == {"from": "n1", "to": "n2"}
        assert all(set(n) == {"id", "type", "params"} for n in arch["nodes"])
        assert c.n == before  # read does not broadcast


# --------------------------------------------------------------------------- #
# validate_architecture (read, no broadcast — runs real execution)
# --------------------------------------------------------------------------- #
@asynctest
async def test_validate_valid_cnn():
    async with harness() as (session, _store, c):
        await _build_cnn(session)
        before = c.n
        res = ok(await session.call_tool("validate_architecture", {}))
        assert res["valid"] is True
        assert res["output_shapes"] == [[2, 10]]
        assert res["output_node_ids"] == ["n5"]
        assert c.n == before  # read does not broadcast


@asynctest
async def test_validate_channel_mismatch_names_failing_node():
    async with harness() as (session, _store, _c):
        n1 = await _add(session, "input", {"shape": [1, 28, 28], "dtype": "float32"})
        n2 = await _add(session, "conv2d",
                        {"in_channels": 1, "out_channels": 16, "kernel_size": 3})
        n3 = await _add(session, "conv2d",
                        {"in_channels": 8, "out_channels": 32, "kernel_size": 3})
        ok(await session.call_tool("connect_layers", {"from_id": n1, "to_id": n2}))
        ok(await session.call_tool("connect_layers", {"from_id": n2, "to_id": n3}))
        res = ok(await session.call_tool("validate_architecture", {}))
        assert res["valid"] is False
        assert res["error"]["node_id"] == n3
        assert res["error"]["layer_type"] == "conv2d"


@asynctest
async def test_validate_no_input_is_valid_false_not_error():
    async with harness() as (session, _store, _c):
        # validate on an empty graph is a successful tool call reporting valid:false
        res = ok(await session.call_tool("validate_architecture", {}))
        assert res["valid"] is False
        assert "input" in res["error"]["message"].lower()


# --------------------------------------------------------------------------- #
# generate_code (read, no broadcast)
# --------------------------------------------------------------------------- #
@asynctest
async def test_generate_code_returns_self_contained_source():
    async with harness() as (session, _store, c):
        await _build_cnn(session)
        before = c.n
        res = ok(await session.call_tool("generate_code", {}))
        code = res["code"]
        assert "class " in code and "def forward" in code
        assert "class LayerExecutionError" in code   # inlined, self-contained
        assert "import backend" not in code          # no project imports
        assert c.n == before  # read does not broadcast


@asynctest
async def test_generate_code_without_input_errors():
    async with harness() as (session, _store, _c):
        n = await _add(session, "relu")
        assert n  # node exists but no input node
        msg = err(await session.call_tool("generate_code", {}))
        assert "input" in msg.lower()


# --------------------------------------------------------------------------- #
# reset_architecture
# --------------------------------------------------------------------------- #
@asynctest
async def test_reset_clears_and_restarts_ids():
    async with harness() as (session, _store, c):
        await _build_cnn(session)
        before = c.n
        assert ok(await session.call_tool("reset_architecture", {})) == {"reset": True}
        assert c.n == before + 1
        arch = ok(await session.call_tool("get_architecture", {}))
        assert arch == {"nodes": [], "edges": []}
        # id counter restarts at n1
        assert await _add(session, "input", {"shape": [1, 8, 8], "dtype": "float32"}) == "n1"


# --------------------------------------------------------------------------- #
# full round-trip
# --------------------------------------------------------------------------- #
@asynctest
async def test_round_trip_build_validate_generate_reset():
    async with harness() as (session, _store, _c):
        await _build_cnn(session)
        assert ok(await session.call_tool("validate_architecture", {}))["valid"] is True
        assert ok(await session.call_tool("generate_code", {}))["code"]
        ok(await session.call_tool("reset_architecture", {}))
        assert ok(await session.call_tool("get_architecture", {}))["nodes"] == []
