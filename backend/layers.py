"""Layer catalog: schema + structural validation ONLY.

This module is the single place that knows which layer types exist and what
params they take. It performs *structural* validation only (unknown type,
missing required param, unknown param) — never tensor-shape math. Shape
correctness is determined exclusively by real execution in ``validator.py``.

Each catalog entry carries:
  - ``category``: frontend color-coding hint (not enforced anywhere).
  - ``required``: list of param names that must be supplied.
  - ``optional``: dict of param name -> default value (applied when omitted).

How each type turns into PyTorch code lives in ``codegen.py``, not here.
Adding a layer = one entry here + one constructor mapping in codegen.py.
"""

# type -> {category, required: [..], optional: {name: default}}
CATALOG = {
    # --- input ---
    "input": {"category": "input", "required": ["shape", "dtype"], "optional": {}},

    # --- conv ---
    "conv2d": {"category": "conv", "required": ["in_channels", "out_channels", "kernel_size"],
               "optional": {"stride": 1, "padding": 0, "dilation": 1}},
    "conv1d": {"category": "conv", "required": ["in_channels", "out_channels", "kernel_size"],
               "optional": {"stride": 1, "padding": 0, "dilation": 1}},
    "conv3d": {"category": "conv", "required": ["in_channels", "out_channels", "kernel_size"],
               "optional": {"stride": 1, "padding": 0, "dilation": 1}},
    "conv_transpose2d": {"category": "conv",
                         "required": ["in_channels", "out_channels", "kernel_size"],
                         "optional": {"stride": 1, "padding": 0, "output_padding": 0, "dilation": 1}},

    # --- pooling / resampling ---
    "maxpool2d": {"category": "pooling", "required": ["kernel_size"],
                  "optional": {"stride": None, "padding": 0}},
    "avgpool2d": {"category": "pooling", "required": ["kernel_size"],
                  "optional": {"stride": None, "padding": 0}},
    "maxpool1d": {"category": "pooling", "required": ["kernel_size"],
                  "optional": {"stride": None, "padding": 0}},
    "avgpool1d": {"category": "pooling", "required": ["kernel_size"],
                  "optional": {"stride": None, "padding": 0}},
    "adaptive_avg_pool2d": {"category": "pooling", "required": ["output_size"], "optional": {}},
    "adaptive_max_pool2d": {"category": "pooling", "required": ["output_size"], "optional": {}},
    "upsample": {"category": "pooling", "required": [],
                 "optional": {"scale_factor": 2, "mode": "nearest"}},

    # --- norm ---
    "batchnorm2d": {"category": "norm", "required": ["num_features"], "optional": {}},
    "batchnorm1d": {"category": "norm", "required": ["num_features"], "optional": {}},
    "groupnorm": {"category": "norm", "required": ["num_groups", "num_channels"], "optional": {}},
    "instancenorm2d": {"category": "norm", "required": ["num_features"], "optional": {}},
    "layernorm": {"category": "norm", "required": ["normalized_shape"], "optional": {}},

    # --- activation ---
    "relu": {"category": "activation", "required": [], "optional": {}},
    "gelu": {"category": "activation", "required": [], "optional": {}},
    "sigmoid": {"category": "activation", "required": [], "optional": {}},
    "tanh": {"category": "activation", "required": [], "optional": {}},
    "leaky_relu": {"category": "activation", "required": [], "optional": {"negative_slope": 0.01}},
    "elu": {"category": "activation", "required": [], "optional": {"alpha": 1.0}},
    "softmax": {"category": "activation", "required": ["dim"], "optional": {}},

    # --- shape / dense / regularization / embedding ---
    "flatten": {"category": "shape", "required": [], "optional": {"start_dim": 1}},
    "linear": {"category": "linear", "required": ["in_features", "out_features"], "optional": {}},
    "dropout": {"category": "regularization", "required": [], "optional": {"p": 0.5}},
    "embedding": {"category": "embedding", "required": ["num_embeddings", "embedding_dim"],
                  "optional": {}},

    # --- recurrent ---
    "rnn": {"category": "recurrent", "required": ["input_size", "hidden_size"],
            "optional": {"num_layers": 1, "bidirectional": False}},
    "lstm": {"category": "recurrent", "required": ["input_size", "hidden_size"],
             "optional": {"num_layers": 1, "bidirectional": False}},
    "gru": {"category": "recurrent", "required": ["input_size", "hidden_size"],
            "optional": {"num_layers": 1, "bidirectional": False}},

    # --- attention / transformer ---
    "positional_encoding": {"category": "attention", "required": ["d_model"],
                            "optional": {"max_len": 5000}},
    "multihead_attention": {"category": "attention", "required": ["embed_dim", "num_heads"],
                            "optional": {"dropout": 0.0}},
    "transformer_encoder_layer": {"category": "attention", "required": ["d_model", "nhead"],
                                  "optional": {"dim_feedforward": 2048, "dropout": 0.1}},

    # --- merge (forward-only, no nn.Module) ---
    "add": {"category": "merge", "required": [], "optional": {}},
    "concat": {"category": "merge", "required": ["dim"], "optional": {}},
}

MERGE_TYPES = {"add", "concat"}


def validate_and_merge(layer_type, params):
    """Structurally validate ``params`` for ``layer_type`` and return a merged
    dict with all defaults applied. Raises ``ValueError`` on any structural
    problem. No tensor-shape checking happens here.
    """
    if layer_type not in CATALOG:
        raise ValueError(f"unknown layer type '{layer_type}'")
    spec = CATALOG[layer_type]

    merged = {}
    for name in spec["required"]:
        if name not in params:
            raise ValueError(f"layer '{layer_type}' missing required param '{name}'")
        merged[name] = params[name]
    for name, default in spec["optional"].items():
        merged[name] = params.get(name, default)

    known = set(spec["required"]) | set(spec["optional"])
    unknown = sorted(set(params) - known)
    if unknown:
        raise ValueError(f"layer '{layer_type}' got unknown param(s): {unknown}")
    return merged
