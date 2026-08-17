"""Layer catalog: schema + structural/value validation ONLY.

This module is the single place that knows which layer types exist and what
params they take. It performs *structural* validation (unknown type, missing
required param, unknown param) plus per-param *value* checks (right type,
sane bounds - see ``_VALUE_SPECS``) - never tensor-shape math. Shape
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


# --- per-param value specs -----------------------------------------------------
#
# Structural checking above answers "which params exist"; this table answers
# "what values does each accept". It is deliberately a flat name->spec table with
# one checker, not a validation framework: its job is to keep params *honest
# scalars* - a string "100000" or a 10**12 int must not slip past the validator's
# pre-flight param estimate (whose arithmetic silently returns 0 on a TypeError)
# - never to do tensor-shape math (invariant 4: shapes are decided only by real
# execution in validator.py). Param names are consistent across layer types, so
# one entry per name covers every type that uses it.
#
# Spec kinds: ("int", lo, hi) · ("int_or_none", lo, hi) · ("int_or_ints", lo, hi)
# int or list/tuple of 1-4 such ints · ("shape",) list of 1-4 ints ·
# ("float", lo, hi) · ("bool",) · ("enum", (values...,))

_INT_MAX = 1_000_000_000

_VALUE_SPECS = {
    "shape": ("shape",),
    "dtype": ("enum", ("float32", "int64")),
    "mode": ("enum", ("nearest", "bilinear", "bicubic")),
    # positive integer sizes/counts
    "in_channels": ("int", 1, _INT_MAX), "out_channels": ("int", 1, _INT_MAX),
    "num_features": ("int", 1, _INT_MAX), "num_groups": ("int", 1, _INT_MAX),
    "num_channels": ("int", 1, _INT_MAX), "in_features": ("int", 1, _INT_MAX),
    "out_features": ("int", 1, _INT_MAX), "num_embeddings": ("int", 1, _INT_MAX),
    "embedding_dim": ("int", 1, _INT_MAX), "input_size": ("int", 1, _INT_MAX),
    "hidden_size": ("int", 1, _INT_MAX), "d_model": ("int", 1, _INT_MAX),
    "nhead": ("int", 1, _INT_MAX), "embed_dim": ("int", 1, _INT_MAX),
    "num_heads": ("int", 1, _INT_MAX), "dim_feedforward": ("int", 1, _INT_MAX),
    "max_len": ("int", 1, _INT_MAX), "num_layers": ("int", 1, 64),
    "dilation": ("int", 1, _INT_MAX),
    "padding": ("int", 0, _INT_MAX), "output_padding": ("int", 0, _INT_MAX),
    # int-or-tuple geometry params
    "kernel_size": ("int_or_ints", 1, _INT_MAX),
    "output_size": ("int_or_ints", 1, _INT_MAX),
    "normalized_shape": ("int_or_ints", 1, _INT_MAX),
    # stride=None means "default to kernel_size" for the pool layers
    "stride": ("int_or_none", 1, _INT_MAX),
    # small signed ints (dim indices)
    "dim": ("int", -8, 8), "start_dim": ("int", -8, 8),
    # probabilities and other floats
    "p": ("float", 0.0, 1.0), "dropout": ("float", 0.0, 1.0),
    "negative_slope": ("float", -1e6, 1e6), "alpha": ("float", -1e6, 1e6),
    "scale_factor": ("float", 1e-6, 1e6),
    "bidirectional": ("bool",),
}


def _is_int(value):
    # bool is an int subclass; True/False must not pass as 1/0
    return isinstance(value, int) and not isinstance(value, bool)


def _check_value(layer_type, name, value):
    """Raise ValueError if ``value`` is out of spec for param ``name``. Error
    style matches the structural errors above: one actionable sentence naming
    the layer, the param, what was expected, and what arrived."""
    spec = _VALUE_SPECS[name]
    kind = spec[0]

    def fail(expected):
        raise ValueError(
            f"layer '{layer_type}' param '{name}': expected {expected}, got {value!r}")

    if kind == "int":
        lo, hi = spec[1], spec[2]
        if not _is_int(value) or not lo <= value <= hi:
            fail(f"an integer in [{lo}, {hi}]")
    elif kind == "int_or_none":
        lo, hi = spec[1], spec[2]
        if value is not None and (not _is_int(value) or not lo <= value <= hi):
            fail(f"an integer in [{lo}, {hi}] or null")
    elif kind == "int_or_ints":
        lo, hi = spec[1], spec[2]
        ok = (_is_int(value) and lo <= value <= hi) or (
            isinstance(value, (list, tuple)) and 1 <= len(value) <= 4
            and all(_is_int(v) and lo <= v <= hi for v in value))
        if not ok:
            fail(f"an integer in [{lo}, {hi}] or a list of 1-4 such integers")
    elif kind == "shape":
        ok = (isinstance(value, (list, tuple)) and 1 <= len(value) <= 4
              and all(_is_int(v) and 1 <= v <= _INT_MAX for v in value))
        if not ok:
            fail("a list of 1-4 positive integers (per-sample shape, no batch dim)")
    elif kind == "float":
        lo, hi = spec[1], spec[2]
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not lo <= value <= hi:
            fail(f"a number in [{lo}, {hi}]")
    elif kind == "bool":
        if not isinstance(value, bool):
            fail("true or false")
    elif kind == "enum":
        if value not in spec[1]:
            fail("one of " + ", ".join(repr(v) for v in spec[1]))


def describe_catalog():
    """Return a JSON-serialisable view of the whole catalog: ``{type: {category,
    required: [...], optional: {name: default}}}``. This is what the agent-facing
    ``get_catalog`` tool exposes so an LLM can discover valid types and params
    instead of probing by trial-and-error.
    """
    return {
        layer_type: {
            "category": spec["category"],
            "required": list(spec["required"]),
            "optional": dict(spec["optional"]),
        }
        for layer_type, spec in CATALOG.items()
    }


def validate_and_merge(layer_type, params):
    """Structurally validate ``params`` for ``layer_type`` and return a merged
    dict with all defaults applied. Raises ``ValueError`` on any structural
    problem. No tensor-shape checking happens here.

    Errors are written for an agent reader: an unknown type lists the known
    types, and a missing-param error names *every* missing required param at once
    (not just the first), so the caller fixes one call instead of looping.
    """
    if layer_type not in CATALOG:
        known = ", ".join(sorted(CATALOG))
        raise ValueError(f"unknown layer type '{layer_type}'; known types: {known}")
    spec = CATALOG[layer_type]

    missing = [name for name in spec["required"] if name not in params]
    if missing:
        raise ValueError(
            f"layer '{layer_type}' missing required param(s): {missing}; "
            f"required: {list(spec['required'])}")

    merged = {name: params[name] for name in spec["required"]}
    for name, default in spec["optional"].items():
        merged[name] = params.get(name, default)

    known = set(spec["required"]) | set(spec["optional"])
    unknown = sorted(set(params) - known)
    if unknown:
        raise ValueError(
            f"layer '{layer_type}' got unknown param(s): {unknown}; "
            f"allowed: {sorted(known)}")

    for name, value in merged.items():
        _check_value(layer_type, name, value)
    return merged
