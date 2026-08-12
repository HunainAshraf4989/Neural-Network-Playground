"""Per-param VALUE validation in layers.validate_and_merge.

Structural checks (unknown type/param, missing required) are covered by the
store tests; these cover the value layer: wrong types (the "100000"-as-string
case that used to slip past the validator's param-count preflight), bool
masquerading as int, out-of-bounds ints, probability ranges, shape lists, and
the dtype/mode enums. Shape *math* is still validator.py's job — nothing here
asserts tensor shapes.
"""

import pytest

import layers


# -- values that must be accepted (incl. every catalog default) -----------------

def test_valid_params_and_defaults_pass():
    merged = layers.validate_and_merge(
        "conv2d", {"in_channels": 1, "out_channels": 16, "kernel_size": 3})
    assert merged["stride"] == 1  # defaults applied and accepted

    # every catalog entry's defaults must survive the value checks
    for layer_type, spec in layers.CATALOG.items():
        params = {}
        if layer_type == "input":
            params = {"shape": [1, 28, 28], "dtype": "float32"}
        elif layer_type == "conv2d" or layer_type in ("conv1d", "conv3d", "conv_transpose2d"):
            params = {"in_channels": 1, "out_channels": 8, "kernel_size": 3}
        elif "kernel_size" in spec["required"]:
            params = {"kernel_size": 2}
        elif layer_type in ("adaptive_avg_pool2d", "adaptive_max_pool2d"):
            params = {"output_size": [1, 1]}
        elif layer_type == "groupnorm":
            params = {"num_groups": 4, "num_channels": 16}
        elif layer_type == "layernorm":
            params = {"normalized_shape": 128}
        elif layer_type in ("batchnorm1d", "batchnorm2d", "instancenorm2d"):
            params = {"num_features": 16}
        elif layer_type == "softmax":
            params = {"dim": -1}
        elif layer_type == "linear":
            params = {"in_features": 64, "out_features": 10}
        elif layer_type == "embedding":
            params = {"num_embeddings": 1000, "embedding_dim": 64}
        elif layer_type in ("rnn", "lstm", "gru"):
            params = {"input_size": 64, "hidden_size": 128}
        elif layer_type == "positional_encoding":
            params = {"d_model": 128}
        elif layer_type == "multihead_attention":
            params = {"embed_dim": 128, "num_heads": 8}
        elif layer_type == "transformer_encoder_layer":
            params = {"d_model": 128, "nhead": 8}
        elif layer_type == "concat":
            params = {"dim": 1}
        layers.validate_and_merge(layer_type, params)  # must not raise


def test_int_or_tuple_geometry_params_accept_lists():
    layers.validate_and_merge(
        "conv2d", {"in_channels": 1, "out_channels": 8, "kernel_size": [3, 5]})
    layers.validate_and_merge("layernorm", {"normalized_shape": [4, 8]})


def test_pool_stride_none_is_allowed():
    merged = layers.validate_and_merge("maxpool2d", {"kernel_size": 2, "stride": None})
    assert merged["stride"] is None


# -- values that must be rejected ------------------------------------------------

@pytest.mark.parametrize("layer_type,params,needle", [
    # a numeric string must not pass as an int (used to bypass the param-count
    # preflight via its except TypeError: return 0 and fall through to RLIMIT)
    ("linear", {"in_features": "100000", "out_features": 10}, "in_features"),
    # bool is an int subclass; True must not pass as 1
    ("conv2d", {"in_channels": True, "out_channels": 8, "kernel_size": 3}, "in_channels"),
    # bounds
    ("linear", {"in_features": 0, "out_features": 10}, "in_features"),
    ("linear", {"in_features": -5, "out_features": 10}, "in_features"),
    ("linear", {"in_features": 10**12, "out_features": 10}, "in_features"),
    ("conv2d", {"in_channels": 1, "out_channels": 8, "kernel_size": 0}, "kernel_size"),
    ("conv2d", {"in_channels": 1, "out_channels": 8, "kernel_size": 3, "padding": -1}, "padding"),
    ("maxpool2d", {"kernel_size": 2, "stride": 0}, "stride"),
    # probabilities live in [0, 1]
    ("dropout", {"p": 1.5}, "'p'"),
    ("dropout", {"p": -0.1}, "'p'"),
    ("dropout", {"p": "0.5"}, "'p'"),
    ("multihead_attention", {"embed_dim": 8, "num_heads": 2, "dropout": 2.0}, "dropout"),
    # shape: list of 1-4 positive ints
    ("input", {"shape": "1,28,28", "dtype": "float32"}, "shape"),
    ("input", {"shape": [], "dtype": "float32"}, "shape"),
    ("input", {"shape": [1, 2, 3, 4, 5], "dtype": "float32"}, "shape"),
    ("input", {"shape": [1, -1], "dtype": "float32"}, "shape"),
    ("input", {"shape": [1, 2.5], "dtype": "float32"}, "shape"),
    # enums
    ("input", {"shape": [1, 4], "dtype": "float64"}, "dtype"),
    ("upsample", {"mode": "warp"}, "mode"),
    # booleans must be real booleans
    ("rnn", {"input_size": 4, "hidden_size": 8, "bidirectional": 1}, "bidirectional"),
    ("rnn", {"input_size": 4, "hidden_size": 8, "bidirectional": "yes"}, "bidirectional"),
    # geometry lists: bad element / too long
    ("conv2d", {"in_channels": 1, "out_channels": 8, "kernel_size": [3, "3"]}, "kernel_size"),
    ("conv2d", {"in_channels": 1, "out_channels": 8, "kernel_size": [1, 1, 1, 1, 1]}, "kernel_size"),
    # dim indices are small signed ints
    ("softmax", {"dim": 99}, "'dim'"),
    ("concat", {"dim": "last"}, "'dim'"),
])
def test_rejects_bad_value(layer_type, params, needle):
    with pytest.raises(ValueError) as exc:
        layers.validate_and_merge(layer_type, params)
    msg = str(exc.value)
    assert needle in msg
    assert f"layer '{layer_type}'" in msg  # error style matches structural errors


def test_error_message_names_expectation_and_actual():
    with pytest.raises(ValueError) as exc:
        layers.validate_and_merge("linear", {"in_features": "64", "out_features": 10})
    assert "expected" in str(exc.value) and "'64'" in str(exc.value)
