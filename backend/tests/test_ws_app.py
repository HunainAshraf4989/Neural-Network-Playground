"""Rigorous coverage of the websocket adapter (Stage 4).

These run the FastAPI app in-process via Starlette's ``TestClient`` against a real
``ArchitectureStore`` (no MCP, no subprocess) and exercise every branch of the
§12 protocol: initial state on connect, ack + broadcast on each of the six
mutations, an ``error`` reply (and *no* broadcast) for every rejection branch,
multi-client broadcast fan-out, and the malformed-input cases (bad json,
non-object, unknown type, missing field) that must never drop the socket.

The cross-surface guarantee (an MCP mutation broadcasts to ws clients, and both
surfaces share one store) is proved end-to-end over the real transport in
``test_ws_stdio.py``; here we prove the ws side in isolation.
"""

import json

from fastapi.testclient import TestClient

from store import ArchitectureStore
from ws_app import build_app


def _client():
    app, _broadcast = build_app(ArchitectureStore())
    return TestClient(app)


def _recv(ws):
    return json.loads(ws.receive_text())


def _add(ws, layer_type, params=None):
    """Send add_layer, drain ack + broadcast, return the broadcast state."""
    ws.send_json({"type": "add_layer", "layer_type": layer_type, "params": params or {}})
    ack = _recv(ws)
    assert ack == {"type": "ack", "ok": True}
    state = _recv(ws)
    assert state["type"] == "state"
    return state["data"]


# -- connect ---------------------------------------------------------------

def test_initial_state_on_connect_is_empty():
    with _client() as c, c.websocket_connect("/ws") as ws:
        state = _recv(ws)
        assert state == {"type": "state", "data": {"nodes": [], "edges": [], "layout": {}}}


# -- the six mutations: happy path -> ack then broadcast -------------------

def test_add_layer_acks_then_broadcasts_new_node():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)  # initial state
        data = _add(ws, "input", {"shape": [1, 8, 8], "dtype": "float32"})
        assert [n["id"] for n in data["nodes"]] == ["n1"]
        assert data["nodes"][0]["type"] == "input"


def test_update_layer_acks_then_broadcasts():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _add(ws, "input", {"shape": [1, 8, 8], "dtype": "float32"})
        _add(ws, "conv2d", {"in_channels": 1, "out_channels": 8, "kernel_size": 3})
        ws.send_json({"type": "update_layer", "node_id": "n2",
                      "params": {"out_channels": 16}})
        assert _recv(ws) == {"type": "ack", "ok": True}
        data = _recv(ws)["data"]
        n2 = next(n for n in data["nodes"] if n["id"] == "n2")
        assert n2["params"]["out_channels"] == 16


def test_connect_layers_acks_then_broadcasts_edge():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _add(ws, "input", {"shape": [1, 8, 8], "dtype": "float32"})
        _add(ws, "flatten")
        ws.send_json({"type": "connect_layers", "from_id": "n1", "to_id": "n2"})
        assert _recv(ws) == {"type": "ack", "ok": True}
        data = _recv(ws)["data"]
        assert data["edges"] == [{"from": "n1", "to": "n2"}]


def test_disconnect_layers_acks_then_broadcasts():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _add(ws, "input", {"shape": [1, 8, 8], "dtype": "float32"})
        _add(ws, "flatten")
        ws.send_json({"type": "connect_layers", "from_id": "n1", "to_id": "n2"})
        _recv(ws); _recv(ws)
        ws.send_json({"type": "disconnect_layers", "from_id": "n1", "to_id": "n2"})
        assert _recv(ws) == {"type": "ack", "ok": True}
        assert _recv(ws)["data"]["edges"] == []


def test_remove_layer_acks_then_broadcasts_and_cascades_edges():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _add(ws, "input", {"shape": [1, 8, 8], "dtype": "float32"})
        _add(ws, "flatten")
        ws.send_json({"type": "connect_layers", "from_id": "n1", "to_id": "n2"})
        _recv(ws); _recv(ws)
        ws.send_json({"type": "remove_layer", "node_id": "n2"})
        assert _recv(ws) == {"type": "ack", "ok": True}
        data = _recv(ws)["data"]
        assert [n["id"] for n in data["nodes"]] == ["n1"]
        assert data["edges"] == []  # touching edge cascaded


def test_reset_architecture_acks_then_broadcasts_empty():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _add(ws, "input", {"shape": [1, 8, 8], "dtype": "float32"})
        ws.send_json({"type": "reset_architecture"})
        assert _recv(ws) == {"type": "ack", "ok": True}
        assert _recv(ws)["data"] == {"nodes": [], "edges": [], "layout": {}}


def test_ungroup_acks_then_broadcasts_subgraph():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _add(ws, "input", {"shape": [10, 32], "dtype": "float32"})
        _add(ws, "transformer_encoder_layer", {"d_model": 32, "nhead": 4})
        ws.send_json({"type": "connect_layers", "from_id": "n1", "to_id": "n2"})
        _recv(ws); _recv(ws)
        ws.send_json({"type": "ungroup", "node_id": "n2"})
        assert _recv(ws) == {"type": "ack", "ok": True}
        data = _recv(ws)["data"]
        ids = [n["id"] for n in data["nodes"]]
        assert "n2" not in ids
        assert len(ids) == 1 + 11  # input + the transformer's 11 sub-layers
        # the input was rewired into both entries (self-attn + residual add)
        types = {n["id"]: n["type"] for n in data["nodes"]}
        entry_types = sorted(types[e["to"]] for e in data["edges"] if e["from"] == "n1")
        assert entry_types == ["add", "multihead_attention"]


# -- generate_code: read-only reply, NO broadcast --------------------------

def test_generate_code_returns_source_without_broadcast():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _add(ws, "input", {"shape": [1, 8, 8], "dtype": "float32"})
        ws.send_json({"type": "generate_code"})
        reply = _recv(ws)
        assert reply["type"] == "code"
        assert "import torch" in reply["code"]
        # no broadcast followed: a subsequent op's first reply is its own ack
        ws.send_json({"type": "add_layer", "layer_type": "flatten", "params": {}})
        assert _recv(ws) == {"type": "ack", "ok": True}
        _recv(ws)  # drain that op's broadcast


def test_generate_code_with_no_input_node_returns_error_field():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        ws.send_json({"type": "generate_code"})
        reply = _recv(ws)
        assert reply["type"] == "code"
        assert "error" in reply and "code" not in reply


# -- rejection branches: error reply, NO broadcast -------------------------

def _assert_error_no_broadcast(ws, msg, needle=None):
    ws.send_json(msg)
    reply = _recv(ws)
    assert reply["type"] == "error", reply
    if needle:
        assert needle in reply["message"], reply["message"]
    # prove no broadcast followed: a subsequent valid op's first reply is its ack
    ws.send_json({"type": "add_layer", "layer_type": "flatten", "params": {}})
    assert _recv(ws) == {"type": "ack", "ok": True}
    _recv(ws)  # drain that op's broadcast


def test_add_layer_unknown_type_errors_without_broadcast():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _assert_error_no_broadcast(ws, {"type": "add_layer", "layer_type": "bogus",
                                        "params": {}})


def test_second_input_node_errors():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _add(ws, "input", {"shape": [1, 8, 8], "dtype": "float32"})
        _assert_error_no_broadcast(
            ws, {"type": "add_layer", "layer_type": "input",
                 "params": {"shape": [1, 4, 4], "dtype": "float32"}},
            needle="input node already exists")


def test_update_unknown_node_errors():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _assert_error_no_broadcast(ws, {"type": "update_layer", "node_id": "n99",
                                        "params": {}}, needle="unknown node")


def test_remove_unknown_node_errors():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _assert_error_no_broadcast(ws, {"type": "remove_layer", "node_id": "n99"},
                                   needle="unknown node")


def test_connect_self_loop_errors():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _add(ws, "input", {"shape": [1, 8, 8], "dtype": "float32"})
        _assert_error_no_broadcast(ws, {"type": "connect_layers", "from_id": "n1",
                                        "to_id": "n1"}, needle="self-loop")


def test_connect_unknown_node_errors():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _add(ws, "input", {"shape": [1, 8, 8], "dtype": "float32"})
        _assert_error_no_broadcast(ws, {"type": "connect_layers", "from_id": "n1",
                                        "to_id": "n42"}, needle="unknown target")


def test_connect_cycle_errors():
    # Use merge nodes so the back-edge clears the single-input guard and actually
    # reaches the cycle branch (n2 -> n3 already exists; n3 -> n2 closes the loop).
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _add(ws, "input", {"shape": [1, 8, 8], "dtype": "float32"})
        _add(ws, "add")
        _add(ws, "add")
        ws.send_json({"type": "connect_layers", "from_id": "n2", "to_id": "n3"})
        _recv(ws); _recv(ws)
        _assert_error_no_broadcast(ws, {"type": "connect_layers", "from_id": "n3",
                                        "to_id": "n2"}, needle="cycle")


def test_connect_duplicate_edge_errors():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _add(ws, "input", {"shape": [1, 8, 8], "dtype": "float32"})
        _add(ws, "flatten")
        ws.send_json({"type": "connect_layers", "from_id": "n1", "to_id": "n2"})
        _recv(ws); _recv(ws)
        _assert_error_no_broadcast(ws, {"type": "connect_layers", "from_id": "n1",
                                        "to_id": "n2"}, needle="already exists")


def test_connect_second_input_into_non_merge_errors():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _add(ws, "input", {"shape": [1, 8, 8], "dtype": "float32"})
        _add(ws, "flatten")   # non-merge target
        _add(ws, "dropout")
        ws.send_json({"type": "connect_layers", "from_id": "n1", "to_id": "n2"})
        _recv(ws); _recv(ws)
        _assert_error_no_broadcast(ws, {"type": "connect_layers", "from_id": "n3",
                                        "to_id": "n2"}, needle="already has an incoming edge")


def test_ungroup_non_composite_errors():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _add(ws, "relu")
        _assert_error_no_broadcast(ws, {"type": "ungroup", "node_id": "n1"},
                                   needle="cannot ungroup")


def test_disconnect_nonexistent_edge_errors():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _add(ws, "input", {"shape": [1, 8, 8], "dtype": "float32"})
        _add(ws, "flatten")
        _assert_error_no_broadcast(ws, {"type": "disconnect_layers", "from_id": "n1",
                                        "to_id": "n2"}, needle="does not exist")


# -- malformed input: error reply, socket stays alive ----------------------

def test_unknown_message_type_errors():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        _assert_error_no_broadcast(ws, {"type": "frobnicate"},
                                   needle="unknown message type")


def test_missing_required_field_errors():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        # add_layer with no layer_type
        _assert_error_no_broadcast(ws, {"type": "add_layer", "params": {}},
                                   needle="missing required field")


def test_malformed_json_errors_without_dropping_socket():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        ws.send_text("{not valid json")
        reply = _recv(ws)
        assert reply["type"] == "error" and "malformed json" in reply["message"]
        # socket still usable
        data = _add(ws, "flatten")
        assert [n["id"] for n in data["nodes"]] == ["n1"]


def test_non_object_message_errors():
    with _client() as c, c.websocket_connect("/ws") as ws:
        _recv(ws)
        ws.send_text("[1, 2, 3]")
        reply = _recv(ws)
        assert reply["type"] == "error" and "json object" in reply["message"]


# A well-formed JSON message whose ``params`` is not an object (null / list /
# string) must surface as a clean ``error`` - never escape as a TypeError and
# drop the socket. The store guards this so both surfaces stay protected.

def test_add_layer_non_dict_params_errors_without_dropping_socket():
    for bad in (None, [1, 2, 3], "x"):
        with _client() as c, c.websocket_connect("/ws") as ws:
            _recv(ws)
            _assert_error_no_broadcast(
                ws, {"type": "add_layer", "layer_type": "flatten", "params": bad},
                needle="params")


def test_update_layer_non_dict_params_errors_without_dropping_socket():
    for bad in (None, [1, 2, 3], "x"):
        with _client() as c, c.websocket_connect("/ws") as ws:
            _recv(ws)
            _add(ws, "flatten")
            _assert_error_no_broadcast(
                ws, {"type": "update_layer", "node_id": "n1", "params": bad},
                needle="params")


# -- multi-client broadcast fan-out ----------------------------------------

def test_mutation_broadcasts_to_all_clients():
    with _client() as c:
        with c.websocket_connect("/ws") as a, c.websocket_connect("/ws") as b:
            _recv(a)  # a's initial state
            _recv(b)  # b's initial state
            a.send_json({"type": "add_layer", "layer_type": "input",
                         "params": {"shape": [1, 8, 8], "dtype": "float32"}})
            assert _recv(a) == {"type": "ack", "ok": True}  # ack to originator only
            # both clients receive the broadcast state
            for ws in (a, b):
                data = _recv(ws)["data"]
                assert [n["id"] for n in data["nodes"]] == ["n1"]


def test_new_client_sees_existing_state_on_connect():
    with _client() as c:
        with c.websocket_connect("/ws") as a:
            _recv(a)
            a.send_json({"type": "add_layer", "layer_type": "input",
                         "params": {"shape": [1, 8, 8], "dtype": "float32"}})
            _recv(a); _recv(a)
            # a second client joining should immediately get the current graph
            with c.websocket_connect("/ws") as b:
                state = _recv(b)
                assert [n["id"] for n in state["data"]["nodes"]] == ["n1"]
