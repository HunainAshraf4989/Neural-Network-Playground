"""Unit tests for ArchitectureStore — one happy path + every rejection branch
for each of the 9 operations, plus orphan-node tolerance and real-execution
validation (CNN passes, channel-mismatch fails naming the node).

Store methods are async (they guard mutations with an asyncio.Lock). To stay
dependency-free we wrap each test body in ``asyncio.run`` via @asynctest, and
build the store *inside* the coroutine so its Lock binds to that run's loop.
"""

import asyncio
import functools

import pytest

from store import ArchitectureStore


def asynctest(coro_fn):
    """Turn an ``async def test_*`` into a sync test pytest can collect."""
    @functools.wraps(coro_fn)
    def wrapper():
        asyncio.run(coro_fn())
    return wrapper


async def _input(store, shape=(1, 28, 28), dtype="float32"):
    return (await store.add_layer("input", {"shape": list(shape), "dtype": dtype}))["node_id"]


# --------------------------------------------------------------------------- #
# add_layer
# --------------------------------------------------------------------------- #
@asynctest
async def test_add_layer_assigns_incremental_ids():
    s = ArchitectureStore()
    assert (await s.add_layer("input", {"shape": [1, 28, 28], "dtype": "float32"})) == {"node_id": "n1"}
    assert (await s.add_layer("relu", {})) == {"node_id": "n2"}
    assert (await s.add_layer("gelu", {})) == {"node_id": "n3"}


@asynctest
async def test_add_layer_applies_defaults():
    s = ArchitectureStore()
    await s.add_layer("conv2d", {"in_channels": 1, "out_channels": 8, "kernel_size": 3})
    node = (await s.get_architecture())["nodes"][0]
    assert node["params"] == {"in_channels": 1, "out_channels": 8, "kernel_size": 3,
                              "stride": 1, "padding": 0, "dilation": 1}


@asynctest
async def test_add_layer_unknown_type_rejected():
    s = ArchitectureStore()
    with pytest.raises(ValueError, match="unknown layer type"):
        await s.add_layer("not_a_layer", {})


@asynctest
async def test_add_layer_missing_required_param_rejected():
    s = ArchitectureStore()
    with pytest.raises(ValueError, match="missing required param"):
        await s.add_layer("conv2d", {"in_channels": 1, "out_channels": 8})  # no kernel_size


@asynctest
async def test_add_layer_unknown_param_rejected():
    s = ArchitectureStore()
    with pytest.raises(ValueError, match="unknown param"):
        await s.add_layer("relu", {"inplace": True})


@asynctest
async def test_add_layer_second_input_rejected():
    s = ArchitectureStore()
    await _input(s)
    with pytest.raises(ValueError, match="input node already exists"):
        await s.add_layer("input", {"shape": [3, 8, 8], "dtype": "float32"})
    # the rejected add must not have consumed an id
    assert (await s.add_layer("relu", {}))["node_id"] == "n2"


# --------------------------------------------------------------------------- #
# update_layer
# --------------------------------------------------------------------------- #
@asynctest
async def test_update_layer_partial_merge():
    s = ArchitectureStore()
    nid = (await s.add_layer("conv2d",
                             {"in_channels": 1, "out_channels": 8, "kernel_size": 3}))["node_id"]
    res = await s.update_layer(nid, {"out_channels": 16, "stride": 2})
    assert res["params"]["out_channels"] == 16
    assert res["params"]["stride"] == 2
    assert res["params"]["in_channels"] == 1      # untouched
    assert res["params"]["kernel_size"] == 3      # untouched


@asynctest
async def test_update_layer_missing_node_rejected():
    s = ArchitectureStore()
    with pytest.raises(ValueError, match="unknown node"):
        await s.update_layer("n99", {"p": 0.1})


@asynctest
async def test_update_layer_revalidates_unknown_param():
    s = ArchitectureStore()
    nid = (await s.add_layer("dropout", {}))["node_id"]
    with pytest.raises(ValueError, match="unknown param"):
        await s.update_layer(nid, {"bogus": 1})


# --------------------------------------------------------------------------- #
# remove_layer
# --------------------------------------------------------------------------- #
@asynctest
async def test_remove_layer_cascades_edges():
    s = ArchitectureStore()
    n1 = await _input(s)
    n2 = (await s.add_layer("relu", {}))["node_id"]
    n3 = (await s.add_layer("gelu", {}))["node_id"]
    await s.connect_layers(n1, n2)
    await s.connect_layers(n2, n3)
    res = await s.remove_layer(n2)
    assert res["removed"] is True
    assert {(e["from"], e["to"]) for e in res["edges_removed"]} == {(n1, n2), (n2, n3)}
    state = await s.get_architecture()
    assert [n["id"] for n in state["nodes"]] == [n1, n3]
    assert state["edges"] == []


@asynctest
async def test_remove_layer_missing_node_rejected():
    s = ArchitectureStore()
    with pytest.raises(ValueError, match="unknown node"):
        await s.remove_layer("n42")


# --------------------------------------------------------------------------- #
# connect_layers
# --------------------------------------------------------------------------- #
@asynctest
async def test_connect_layers_happy():
    s = ArchitectureStore()
    n1 = await _input(s)
    n2 = (await s.add_layer("relu", {}))["node_id"]
    assert (await s.connect_layers(n1, n2)) == {"edge": {"from": n1, "to": n2}}


@asynctest
async def test_connect_layers_missing_source_rejected():
    s = ArchitectureStore()
    n2 = (await s.add_layer("relu", {}))["node_id"]
    with pytest.raises(ValueError, match="unknown source node"):
        await s.connect_layers("n99", n2)


@asynctest
async def test_connect_layers_missing_target_rejected():
    s = ArchitectureStore()
    n1 = await _input(s)
    with pytest.raises(ValueError, match="unknown target node"):
        await s.connect_layers(n1, "n99")


@asynctest
async def test_connect_layers_self_loop_rejected():
    s = ArchitectureStore()
    n2 = (await s.add_layer("relu", {}))["node_id"]
    with pytest.raises(ValueError, match="self-loop"):
        await s.connect_layers(n2, n2)


@asynctest
async def test_connect_layers_to_input_rejected():
    s = ArchitectureStore()
    n1 = await _input(s)
    n2 = (await s.add_layer("relu", {}))["node_id"]
    with pytest.raises(ValueError, match="cannot have an incoming edge"):
        await s.connect_layers(n2, n1)


@asynctest
async def test_connect_layers_duplicate_rejected():
    s = ArchitectureStore()
    n1 = await _input(s)
    n2 = (await s.add_layer("relu", {}))["node_id"]
    await s.connect_layers(n1, n2)
    with pytest.raises(ValueError, match="already exists"):
        await s.connect_layers(n1, n2)


@asynctest
async def test_connect_layers_second_inbound_into_non_merge_rejected():
    # spec §8: a non-input, non-merge node takes exactly one incoming edge.
    # Without this guard codegen silently drops the extra predecessor -> canvas/model desync.
    s = ArchitectureStore()
    n1 = await _input(s, shape=(8,))
    n2 = (await s.add_layer("linear", {"in_features": 8, "out_features": 4}))["node_id"]
    n3 = (await s.add_layer("linear", {"in_features": 8, "out_features": 4}))["node_id"]
    await s.connect_layers(n1, n3)
    with pytest.raises(ValueError, match="already has an incoming edge"):
        await s.connect_layers(n2, n3)
    # only the first edge survived
    assert [(e["from"], e["to"]) for e in (await s.get_architecture())["edges"]] == [(n1, n3)]


@asynctest
async def test_connect_layers_multiple_inbound_into_merge_allowed():
    # merge nodes (add/concat) are the only multi-input nodes — must still accept >1 inbound.
    s = ArchitectureStore()
    n1 = await _input(s, shape=(4,))
    n2 = (await s.add_layer("linear", {"in_features": 4, "out_features": 4}))["node_id"]
    n3 = (await s.add_layer("add", {}))["node_id"]
    await s.connect_layers(n1, n2)
    await s.connect_layers(n1, n3)
    await s.connect_layers(n2, n3)   # second inbound into the merge — allowed
    assert [(e["from"], e["to"]) for e in (await s.get_architecture())["edges"]
            if e["to"] == n3] == [(n1, n3), (n2, n3)]


@asynctest
async def test_connect_layers_cycle_rejected():
    # n2 is a merge node so the second-inbound guard doesn't pre-empt the cycle check;
    # n3 -> n2 closes the loop n2 -> n3 -> n2.
    s = ArchitectureStore()
    n1 = await _input(s)
    n2 = (await s.add_layer("add", {}))["node_id"]
    n3 = (await s.add_layer("gelu", {}))["node_id"]
    await s.connect_layers(n1, n2)
    await s.connect_layers(n2, n3)
    with pytest.raises(ValueError, match="would create a cycle"):
        await s.connect_layers(n3, n2)


# --------------------------------------------------------------------------- #
# disconnect_layers
# --------------------------------------------------------------------------- #
@asynctest
async def test_disconnect_layers_happy():
    s = ArchitectureStore()
    n1 = await _input(s)
    n2 = (await s.add_layer("relu", {}))["node_id"]
    await s.connect_layers(n1, n2)
    assert (await s.disconnect_layers(n1, n2)) == {"removed": True}
    assert (await s.get_architecture())["edges"] == []


@asynctest
async def test_disconnect_layers_absent_rejected():
    s = ArchitectureStore()
    n1 = await _input(s)
    n2 = (await s.add_layer("relu", {}))["node_id"]
    with pytest.raises(ValueError, match="does not exist"):
        await s.disconnect_layers(n1, n2)


# --------------------------------------------------------------------------- #
# get_architecture
# --------------------------------------------------------------------------- #
@asynctest
async def test_get_architecture_preserves_edge_insertion_order():
    s = ArchitectureStore()
    n1 = await _input(s)
    n2 = (await s.add_layer("relu", {}))["node_id"]
    n3 = (await s.add_layer("gelu", {}))["node_id"]
    nc = (await s.add_layer("concat", {"dim": 1}))["node_id"]
    # connect n3 before n2 — concat order must follow this call order, not id order
    await s.connect_layers(n1, n2)
    await s.connect_layers(n1, n3)
    await s.connect_layers(n3, nc)
    await s.connect_layers(n2, nc)
    edges_into_concat = [(e["from"], e["to"]) for e in (await s.get_architecture())["edges"]
                         if e["to"] == nc]
    assert edges_into_concat == [(n3, nc), (n2, nc)]


@asynctest
async def test_get_architecture_returns_deep_copy():
    s = ArchitectureStore()
    await _input(s)
    snap = await s.get_architecture()
    snap["nodes"][0]["params"]["dtype"] = "MUTATED"
    snap["nodes"].append({"id": "fake"})
    fresh = await s.get_architecture()
    assert fresh["nodes"][0]["params"]["dtype"] == "float32"
    assert len(fresh["nodes"]) == 1


# --------------------------------------------------------------------------- #
# reset_architecture
# --------------------------------------------------------------------------- #
@asynctest
async def test_reset_architecture_clears_and_resets_counter():
    s = ArchitectureStore()
    await _input(s)
    await s.add_layer("relu", {})
    assert (await s.reset_architecture()) == {"reset": True}
    state = await s.get_architecture()
    assert state == {"nodes": [], "edges": [], "layout": {}}
    assert (await s.add_layer("relu", {}))["node_id"] == "n1"   # counter back to 0


# --------------------------------------------------------------------------- #
# generate_code (delegates to codegen)
# --------------------------------------------------------------------------- #
@asynctest
async def test_generate_code_emits_model_class():
    s = ArchitectureStore()
    n1 = await _input(s)
    n2 = (await s.add_layer("relu", {}))["node_id"]
    await s.connect_layers(n1, n2)
    code = (await s.generate_code())["code"]
    assert "class GeneratedModel(nn.Module):" in code
    assert "import torch" in code


@asynctest
async def test_generate_code_without_input_rejected():
    s = ArchitectureStore()
    await s.add_layer("relu", {})
    with pytest.raises(ValueError, match="no input node"):
        await s.generate_code()


# --------------------------------------------------------------------------- #
# validate_architecture (real subprocess execution — the only shape check)
# --------------------------------------------------------------------------- #
@asynctest
async def test_validate_architecture_cnn_passes():
    s = ArchitectureStore()
    n1 = await _input(s, shape=(1, 28, 28))
    n2 = (await s.add_layer("conv2d",
                            {"in_channels": 1, "out_channels": 16, "kernel_size": 3}))["node_id"]
    n3 = (await s.add_layer("maxpool2d", {"kernel_size": 2}))["node_id"]
    n4 = (await s.add_layer("flatten", {}))["node_id"]
    n5 = (await s.add_layer("linear",
                            {"in_features": 16 * 13 * 13, "out_features": 10}))["node_id"]
    for a, b in [(n1, n2), (n2, n3), (n3, n4), (n4, n5)]:
        await s.connect_layers(a, b)
    res = await s.validate_architecture()
    assert res["valid"] is True
    assert res["output_shapes"] == [[2, 10]]
    assert res["output_node_ids"] == [n5]


@asynctest
async def test_validate_architecture_channel_mismatch_names_node():
    s = ArchitectureStore()
    n1 = await _input(s, shape=(1, 28, 28))
    n2 = (await s.add_layer("conv2d",
                            {"in_channels": 1, "out_channels": 16, "kernel_size": 3}))["node_id"]
    # n3 expects 8 in-channels but receives 16 -> fails at n3
    n3 = (await s.add_layer("conv2d",
                            {"in_channels": 8, "out_channels": 32, "kernel_size": 3}))["node_id"]
    await s.connect_layers(n1, n2)
    await s.connect_layers(n2, n3)
    res = await s.validate_architecture()
    assert res["valid"] is False
    assert res["error"]["node_id"] == n3
    assert res["error"]["layer_type"] == "conv2d"


@asynctest
async def test_validate_architecture_no_input_reports_cleanly():
    s = ArchitectureStore()
    await s.add_layer("relu", {})
    res = await s.validate_architecture()
    assert res["valid"] is False
    assert "input" in res["error"]["message"]


# --------------------------------------------------------------------------- #
# orphan tolerance — an unwired node is allowed, surfaced as a warning
# --------------------------------------------------------------------------- #
@asynctest
async def test_orphan_node_allowed_and_warned():
    s = ArchitectureStore()
    n1 = await _input(s, shape=(1, 28, 28))
    n2 = (await s.add_layer("flatten", {}))["node_id"]
    await s.connect_layers(n1, n2)
    orphan = (await s.add_layer("relu", {}))["node_id"]   # never connected
    res = await s.validate_architecture()
    assert res["valid"] is True
    assert any(orphan in w for w in res["warnings"])


# --------------------------------------------------------------------------- #
# richer errors (agent-facing): list valid types / all missing params at once
# --------------------------------------------------------------------------- #
@asynctest
async def test_unknown_type_error_lists_known_types():
    s = ArchitectureStore()
    with pytest.raises(ValueError, match="known types:") as ei:
        await s.add_layer("conv", {})
    assert "conv2d" in str(ei.value)  # the real type the caller probably meant


@asynctest
async def test_missing_params_error_names_all_missing_at_once():
    s = ArchitectureStore()
    with pytest.raises(ValueError) as ei:
        await s.add_layer("conv2d", {})  # missing in/out_channels AND kernel_size
    msg = str(ei.value)
    assert "missing required param" in msg
    for name in ("in_channels", "out_channels", "kernel_size"):
        assert name in msg


# --------------------------------------------------------------------------- #
# get_catalog
# --------------------------------------------------------------------------- #
@asynctest
async def test_get_catalog_shape():
    s = ArchitectureStore()
    cat = await s.get_catalog()
    assert cat["conv2d"]["category"] == "conv"
    assert "in_channels" in cat["conv2d"]["required"]
    assert cat["conv2d"]["optional"]["stride"] == 1
    assert cat["concat"]["required"] == ["dim"]
    assert "input" in cat


# --------------------------------------------------------------------------- #
# add_layers (atomic batch)
# --------------------------------------------------------------------------- #
@asynctest
async def test_add_layers_happy_returns_ids_in_order():
    s = ArchitectureStore()
    res = await s.add_layers([
        {"type": "input", "params": {"shape": [3, 8, 8], "dtype": "float32"}},
        {"type": "conv2d", "params": {"in_channels": 3, "out_channels": 8, "kernel_size": 3}},
        {"type": "relu", "params": {}},
    ])
    assert res == {"node_ids": ["n1", "n2", "n3"]}
    assert [n["id"] for n in (await s.get_architecture())["nodes"]] == ["n1", "n2", "n3"]


@asynctest
async def test_add_layers_is_atomic_on_failure():
    s = ArchitectureStore()
    await _input(s)  # n1 already exists
    with pytest.raises(ValueError, match="add_layers failed at index 1"):
        await s.add_layers([
            {"type": "relu", "params": {}},          # ok
            {"type": "not_a_layer", "params": {}},   # blows up -> roll back the whole batch
            {"type": "gelu", "params": {}},
        ])
    # nothing from the batch landed, and no ids were consumed past n1
    assert [n["id"] for n in (await s.get_architecture())["nodes"]] == ["n1"]
    assert (await s.add_layer("relu", {}))["node_id"] == "n2"


@asynctest
async def test_add_layers_second_input_in_batch_rolls_back():
    s = ArchitectureStore()
    with pytest.raises(ValueError, match="input node already exists"):
        await s.add_layers([
            {"type": "input", "params": {"shape": [1, 8, 8], "dtype": "float32"}},
            {"type": "input", "params": {"shape": [1, 8, 8], "dtype": "float32"}},
        ])
    assert (await s.get_architecture())["nodes"] == []


@asynctest
async def test_add_layers_rejects_empty():
    s = ArchitectureStore()
    with pytest.raises(ValueError, match="non-empty list"):
        await s.add_layers([])


# --------------------------------------------------------------------------- #
# connect_layers_batch (atomic batch)
# --------------------------------------------------------------------------- #
@asynctest
async def test_connect_layers_batch_happy():
    s = ArchitectureStore()
    ids = (await s.add_layers([
        {"type": "input", "params": {"shape": [1, 8, 8], "dtype": "float32"}},
        {"type": "relu", "params": {}},
        {"type": "flatten", "params": {}},
    ]))["node_ids"]
    res = await s.connect_layers_batch([
        {"from_id": ids[0], "to_id": ids[1]},
        {"from_id": ids[1], "to_id": ids[2]},
    ])
    assert res == {"edges": [{"from": ids[0], "to": ids[1]}, {"from": ids[1], "to": ids[2]}]}


@asynctest
async def test_connect_layers_batch_is_atomic_on_failure():
    s = ArchitectureStore()
    n1 = await _input(s)
    n2 = (await s.add_layer("relu", {}))["node_id"]
    with pytest.raises(ValueError, match="connect_layers_batch failed at index 1"):
        await s.connect_layers_batch([
            {"from_id": n1, "to_id": n2},     # ok
            {"from_id": n2, "to_id": "n99"},  # unknown target -> roll back the batch
        ])
    # the first edge was rolled back too
    assert (await s.get_architecture())["edges"] == []


# --------------------------------------------------------------------------- #
# layout-hint channel (separate from node params; never in nodes)
# --------------------------------------------------------------------------- #
@asynctest
async def test_layout_hint_stored_separately_from_params():
    s = ArchitectureStore()
    nid = (await s.add_layer("relu", {}, layout={"col": 2, "row": 1}))["node_id"]
    arch = await s.get_architecture()
    assert arch["layout"][nid] == {"col": 2, "row": 1}
    # geometry must NOT leak into the node schema
    assert set(arch["nodes"][0]) == {"id", "type", "params"}
    assert "col" not in arch["nodes"][0]["params"]


@asynctest
async def test_layout_hint_dropped_on_remove_and_cleared_on_reset():
    s = ArchitectureStore()
    nid = (await s.add_layer("relu", {}, layout={"col": 1}))["node_id"]
    await s.remove_layer(nid)
    assert (await s.get_architecture())["layout"] == {}
    nid2 = (await s.add_layer("gelu", {}, layout={"row": 3}))["node_id"]
    await s.reset_architecture()
    assert (await s.get_architecture())["layout"] == {}
    assert nid2  # silence unused


# --------------------------------------------------------------------------- #
# ungroup — composite -> real editable sub-layers
# --------------------------------------------------------------------------- #
async def _transformer_chain(s):
    """input(seq) -> transformer_encoder_layer -> linear; returns the three ids."""
    n1 = (await s.add_layer("input", {"shape": [10, 32], "dtype": "float32"}))["node_id"]
    n2 = (await s.add_layer("transformer_encoder_layer",
                            {"d_model": 32, "nhead": 4, "dim_feedforward": 64,
                             "dropout": 0.1}))["node_id"]
    n3 = (await s.add_layer("linear", {"in_features": 32, "out_features": 10}))["node_id"]
    await s.connect_layers(n1, n2)
    await s.connect_layers(n2, n3)
    return n1, n2, n3


@asynctest
async def test_ungroup_transformer_replaces_with_subgraph():
    s = ArchitectureStore()
    n1, n2, n3 = await _transformer_chain(s)
    res = await s.ungroup(n2)
    assert res["ungrouped"] == n2
    ids = res["node_ids"]
    assert set(ids) == {"attn", "do1", "res1", "norm1", "fc1", "act",
                        "do2", "fc2", "do3", "res2", "norm2"}

    arch = await s.get_architecture()
    node_ids = [n["id"] for n in arch["nodes"]]
    assert n2 not in node_ids
    assert len(arch["nodes"]) == 2 + 11  # input + linear + 11 sub-layers

    by_id = {n["id"]: n for n in arch["nodes"]}
    assert by_id[ids["attn"]]["type"] == "multihead_attention"
    assert by_id[ids["attn"]]["params"] == {"embed_dim": 32, "num_heads": 4, "dropout": 0.1}
    assert by_id[ids["fc1"]]["params"] == {"in_features": 32, "out_features": 64}
    assert by_id[ids["fc2"]]["params"] == {"in_features": 64, "out_features": 32}
    assert by_id[ids["norm1"]]["params"] == {"normalized_shape": 32}

    edges = {(e["from"], e["to"]) for e in arch["edges"]}
    # incoming edge fans into BOTH entries (self-attn and the attention residual)
    assert (n1, ids["attn"]) in edges
    assert (n1, ids["res1"]) in edges
    # outgoing edge is rewired from the exit
    assert (ids["norm2"], n3) in edges
    # the feed-forward residual skip exists
    assert (ids["norm1"], ids["res2"]) in edges
    # no edge references the removed composite
    assert not any(n2 in pair for pair in edges)


@asynctest
async def test_ungrouped_transformer_validates_by_real_execution():
    s = ArchitectureStore()
    _n1, n2, _n3 = await _transformer_chain(s)
    await s.ungroup(n2)
    report = await s.validate_architecture()
    assert report["valid"] is True, report
    # (N, L, E) with the fixed N=2 dummy batch, ending in the 10-way linear
    assert report["output_shapes"] == [[2, 10, 10]]


@asynctest
async def test_ungroup_stacked_lstm_chains_cells():
    s = ArchitectureStore()
    n1 = (await s.add_layer("input", {"shape": [10, 16], "dtype": "float32"}))["node_id"]
    n2 = (await s.add_layer("lstm", {"input_size": 16, "hidden_size": 32,
                                     "num_layers": 3}))["node_id"]
    n3 = (await s.add_layer("relu", {}))["node_id"]
    await s.connect_layers(n1, n2)
    await s.connect_layers(n2, n3)
    res = await s.ungroup(n2)
    ids = res["node_ids"]
    assert set(ids) == {"l0", "l1", "l2"}

    arch = await s.get_architecture()
    by_id = {n["id"]: n for n in arch["nodes"]}
    for name in ("l0", "l1", "l2"):
        assert by_id[ids[name]]["type"] == "lstm"
        assert by_id[ids[name]]["params"]["num_layers"] == 1
    assert by_id[ids["l0"]]["params"]["input_size"] == 16
    assert by_id[ids["l1"]]["params"]["input_size"] == 32  # fed layer-0 hidden state
    assert by_id[ids["l2"]]["params"]["input_size"] == 32

    edges = {(e["from"], e["to"]) for e in arch["edges"]}
    assert edges == {(n1, ids["l0"]), (ids["l0"], ids["l1"]),
                     (ids["l1"], ids["l2"]), (ids["l2"], n3)}


@asynctest
async def test_ungroup_bidirectional_gru_splits_and_concats():
    s = ArchitectureStore()
    n1 = (await s.add_layer("input", {"shape": [10, 16], "dtype": "float32"}))["node_id"]
    n2 = (await s.add_layer("gru", {"input_size": 16, "hidden_size": 32,
                                    "bidirectional": True}))["node_id"]
    await s.connect_layers(n1, n2)
    res = await s.ungroup(n2)
    ids = res["node_ids"]
    assert set(ids) == {"l0f", "l0b", "l0c"}

    arch = await s.get_architecture()
    by_id = {n["id"]: n for n in arch["nodes"]}
    assert by_id[ids["l0f"]]["type"] == "gru"
    assert by_id[ids["l0f"]]["params"]["bidirectional"] is False
    assert by_id[ids["l0c"]]["type"] == "concat"
    assert by_id[ids["l0c"]]["params"] == {"dim": -1}

    edges = {(e["from"], e["to"]) for e in arch["edges"]}
    # the input feeds BOTH directions; both feed the concat
    assert edges == {(n1, ids["l0f"]), (n1, ids["l0b"]),
                     (ids["l0f"], ids["l0c"]), (ids["l0b"], ids["l0c"])}


@asynctest
async def test_ungroup_preserves_downstream_concat_input_order():
    # Edge insertion order decides concat argument order, so rewiring must keep
    # the ungrouped branch's edge at its original position in the edge list.
    s = ArchitectureStore()
    n1 = (await s.add_layer("input", {"shape": [10, 16], "dtype": "float32"}))["node_id"]
    n2 = (await s.add_layer("lstm", {"input_size": 16, "hidden_size": 32,
                                     "num_layers": 2}))["node_id"]
    n3 = (await s.add_layer("linear", {"in_features": 16, "out_features": 32}))["node_id"]
    n4 = (await s.add_layer("concat", {"dim": -1}))["node_id"]
    await s.connect_layers(n1, n2)
    await s.connect_layers(n1, n3)
    await s.connect_layers(n2, n4)  # lstm branch is concat's FIRST input
    await s.connect_layers(n3, n4)
    ids = (await s.ungroup(n2))["node_ids"]
    into_concat = [e["from"] for e in (await s.get_architecture())["edges"]
                   if e["to"] == n4]
    assert into_concat == [ids["l1"], n3]  # exit replaced the lstm, order intact


@asynctest
async def test_ungroup_plain_recurrent_rejected():
    s = ArchitectureStore()
    nid = (await s.add_layer("rnn", {"input_size": 8, "hidden_size": 16}))["node_id"]
    with pytest.raises(ValueError, match="no internal structure"):
        await s.ungroup(nid)
    # nothing changed and no ids were consumed
    assert [n["id"] for n in (await s.get_architecture())["nodes"]] == [nid]
    assert (await s.add_layer("relu", {}))["node_id"] == "n2"


@asynctest
async def test_ungroup_non_composite_rejected():
    s = ArchitectureStore()
    nid = (await s.add_layer("relu", {}))["node_id"]
    with pytest.raises(ValueError, match="cannot ungroup"):
        await s.ungroup(nid)


@asynctest
async def test_ungroup_multihead_attention_rejected():
    # Deliberately not ungroupable: its decomposition needs the synthetic
    # scaled_dot_product_attention type that isn't in the catalog yet.
    s = ArchitectureStore()
    nid = (await s.add_layer("multihead_attention",
                             {"embed_dim": 32, "num_heads": 4}))["node_id"]
    with pytest.raises(ValueError, match="cannot ungroup"):
        await s.ungroup(nid)


@asynctest
async def test_ungroup_unknown_node_rejected():
    s = ArchitectureStore()
    with pytest.raises(ValueError, match="unknown node"):
        await s.ungroup("n99")


@asynctest
async def test_ungroup_drops_composite_layout_hint():
    s = ArchitectureStore()
    nid = (await s.add_layer("transformer_encoder_layer",
                             {"d_model": 32, "nhead": 4},
                             layout={"col": 3, "row": 1}))["node_id"]
    await s.ungroup(nid)
    assert (await s.get_architecture())["layout"] == {}


@asynctest
async def test_layout_hint_validation():
    s = ArchitectureStore()
    with pytest.raises(ValueError, match="non-negative integer"):
        await s.add_layer("relu", {}, layout={"col": -1})
    with pytest.raises(ValueError, match="unknown key"):
        await s.add_layer("relu", {}, layout={"x": 5})
    # empty/None hint is fine and simply records nothing
    nid = (await s.add_layer("relu", {}, layout={}))["node_id"]
    assert nid not in (await s.get_architecture())["layout"]
