"""Code generation: turn an AppState graph into a self-contained PyTorch module.

Pipeline:
  1. Find the single ``input`` node; compute the subgraph reachable from it.
  2. Kahn topological sort over that subgraph (ties broken by ascending id for
     deterministic output). Unreachable nodes become non-fatal warnings.
  3. Emit a fully self-contained module string: ``torch``/``nn``/``math`` only,
     inline ``LayerExecutionError``, inline ``_merge_add`` (if any add merge),
     inline ``PositionalEncoding`` (if any positional_encoding), then the model.

The generated file imports NOTHING from this project - copy it next to a fresh
python with only ``torch`` installed and it runs unmodified.
"""

from layers import MERGE_TYPES


# ---------------------------------------------------------------------------
# inline source blocks (kept self-contained; emitted verbatim into the output)
# ---------------------------------------------------------------------------
_LAYER_ERROR_SRC = '''\
class LayerExecutionError(Exception):
    def __init__(self, node_id, layer_type, original):
        self.node_id = node_id
        self.layer_type = layer_type
        self.original = original
        super().__init__("[" + str(node_id) + "] " + str(layer_type) + ": " + repr(original))'''

_MERGE_ADD_SRC = '''\
def _merge_add(tensors):
    base = tensors[0].shape
    for t in tensors[1:]:
        if t.shape != base:
            raise ValueError(
                "add merge requires identical shapes, got "
                + str(tuple(base)) + " and " + str(tuple(t.shape)))
    out = tensors[0]
    for t in tensors[1:]:
        out = out + t
    return out'''

_POS_ENC_SRC = '''\
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]'''


# ---------------------------------------------------------------------------
# per-type constructor mapping: params -> "nn.Xxx(...)" expression string
# (merge/input nodes have no module and are absent from this table)
# ---------------------------------------------------------------------------
def _conv(cls):
    def build(p):
        return (f"nn.{cls}(in_channels={p['in_channels']!r}, out_channels={p['out_channels']!r}, "
                f"kernel_size={p['kernel_size']!r}, stride={p['stride']!r}, "
                f"padding={p['padding']!r}, dilation={p['dilation']!r})")
    return build


def _conv_t(p):
    return (f"nn.ConvTranspose2d(in_channels={p['in_channels']!r}, out_channels={p['out_channels']!r}, "
            f"kernel_size={p['kernel_size']!r}, stride={p['stride']!r}, padding={p['padding']!r}, "
            f"output_padding={p['output_padding']!r}, dilation={p['dilation']!r})")


def _pool(cls):
    def build(p):
        stride = p["stride"] if p["stride"] is not None else p["kernel_size"]
        return f"nn.{cls}(kernel_size={p['kernel_size']!r}, stride={stride!r}, padding={p['padding']!r})"
    return build


def _rnn(cls):
    def build(p):
        return (f"nn.{cls}(input_size={p['input_size']!r}, hidden_size={p['hidden_size']!r}, "
                f"num_layers={p['num_layers']!r}, bidirectional={p['bidirectional']!r}, "
                f"batch_first=True)")
    return build


_CONSTRUCTORS = {
    "conv2d": _conv("Conv2d"),
    "conv1d": _conv("Conv1d"),
    "conv3d": _conv("Conv3d"),
    "conv_transpose2d": _conv_t,
    "maxpool2d": _pool("MaxPool2d"),
    "avgpool2d": _pool("AvgPool2d"),
    "maxpool1d": _pool("MaxPool1d"),
    "avgpool1d": _pool("AvgPool1d"),
    "adaptive_avg_pool2d": lambda p: f"nn.AdaptiveAvgPool2d(output_size={p['output_size']!r})",
    "adaptive_max_pool2d": lambda p: f"nn.AdaptiveMaxPool2d(output_size={p['output_size']!r})",
    "upsample": lambda p: f"nn.Upsample(scale_factor={p['scale_factor']!r}, mode={p['mode']!r})",
    "batchnorm2d": lambda p: f"nn.BatchNorm2d(num_features={p['num_features']!r})",
    "batchnorm1d": lambda p: f"nn.BatchNorm1d(num_features={p['num_features']!r})",
    "groupnorm": lambda p: f"nn.GroupNorm(num_groups={p['num_groups']!r}, num_channels={p['num_channels']!r})",
    "instancenorm2d": lambda p: f"nn.InstanceNorm2d(num_features={p['num_features']!r})",
    "layernorm": lambda p: f"nn.LayerNorm(normalized_shape={p['normalized_shape']!r})",
    "relu": lambda p: "nn.ReLU()",
    "gelu": lambda p: "nn.GELU()",
    "sigmoid": lambda p: "nn.Sigmoid()",
    "tanh": lambda p: "nn.Tanh()",
    "leaky_relu": lambda p: f"nn.LeakyReLU(negative_slope={p['negative_slope']!r})",
    "elu": lambda p: f"nn.ELU(alpha={p['alpha']!r})",
    "softmax": lambda p: f"nn.Softmax(dim={p['dim']!r})",
    "flatten": lambda p: f"nn.Flatten(start_dim={p['start_dim']!r})",
    "linear": lambda p: f"nn.Linear(in_features={p['in_features']!r}, out_features={p['out_features']!r})",
    "dropout": lambda p: f"nn.Dropout(p={p['p']!r})",
    "embedding": lambda p: f"nn.Embedding(num_embeddings={p['num_embeddings']!r}, embedding_dim={p['embedding_dim']!r})",
    "rnn": _rnn("RNN"),
    "lstm": _rnn("LSTM"),
    "gru": _rnn("GRU"),
    "positional_encoding": lambda p: f"PositionalEncoding(d_model={p['d_model']!r}, max_len={p['max_len']!r})",
    "multihead_attention": lambda p: (f"nn.MultiheadAttention(embed_dim={p['embed_dim']!r}, "
                                      f"num_heads={p['num_heads']!r}, dropout={p['dropout']!r}, batch_first=True)"),
    "transformer_encoder_layer": lambda p: (f"nn.TransformerEncoderLayer(d_model={p['d_model']!r}, "
                                            f"nhead={p['nhead']!r}, dim_feedforward={p['dim_feedforward']!r}, "
                                            f"dropout={p['dropout']!r}, batch_first=True)"),
}

# forward-call style for nodes whose module returns more than the bare tensor
_FORWARD_STYLE = {
    "rnn": "rnn", "lstm": "rnn", "gru": "rnn",        # module(x)[0]  -> output, drop hidden
    "multihead_attention": "attention",                # module(x, x, x)[0]  -> self-attention
}


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------
def _idkey(node_id):
    """Sort key for "n<k>" ids; falls back to lexicographic for odd ids."""
    try:
        return (0, int(node_id[1:]))
    except (ValueError, IndexError):
        return (1, node_id)


def _input_id(nodes):
    for nid, n in nodes.items():
        if n["type"] == "input":
            return nid
    raise ValueError("graph has no input node")


def _reachable(input_id, edges):
    adj = {}
    for e in edges:
        adj.setdefault(e["from"], []).append(e["to"])
    seen, stack = set(), [input_id]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(m for m in adj.get(n, []) if m not in seen)
    return seen


def _topo(reachable, edges):
    indeg = {n: 0 for n in reachable}
    adj = {n: [] for n in reachable}
    for e in edges:
        if e["from"] in reachable and e["to"] in reachable:
            adj[e["from"]].append(e["to"])
            indeg[e["to"]] += 1
    ready = sorted((n for n in reachable if indeg[n] == 0), key=_idkey)
    order = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
        ready.sort(key=_idkey)
    return order


def _preds(node_id, edges, reachable):
    """Predecessor ids in edge-insertion order (concat depends on this)."""
    return [e["from"] for e in edges if e["to"] == node_id and e["from"] in reachable]


def _has_succ(node_id, edges):
    return any(e["from"] == node_id for e in edges)


def constructor_code(node):
    """The ``nn.Xxx(...)`` expression for a node, or None for input/merge nodes."""
    t = node["type"]
    if t == "input" or t in MERGE_TYPES:
        return None
    return _CONSTRUCTORS[t](node["params"])


def _forward_rhs(node, preds):
    """Right-hand-side expression assigned to ``outputs[id]`` in forward()."""
    t, nid = node["type"], node["id"]
    if t == "add":
        items = ", ".join(f'outputs["{p}"]' for p in preds)
        return f"_merge_add([{items}])"
    if t == "concat":
        items = ", ".join(f'outputs["{p}"]' for p in preds)
        return f'torch.cat([{items}], dim={node["params"]["dim"]!r})'
    inp = f'outputs["{preds[0]}"]'
    style = _FORWARD_STYLE.get(t, "simple")
    if style == "rnn":
        return f"self.layer_{nid}({inp})[0]"
    if style == "attention":
        return f"self.layer_{nid}({inp}, {inp}, {inp})[0]"
    return f"self.layer_{nid}({inp})"


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def generate(state):
    """Return ``(code, terminal_node_ids, warnings)`` for an AppState graph.

    ``terminal_node_ids`` are the reachable nodes with no outgoing edge, in the
    same ascending-id order the generated ``forward`` returns them. ``warnings``
    lists nodes excluded for being unreachable from the input node.
    """
    nodes = {n["id"]: n for n in state["nodes"]}
    edges = list(state["edges"])

    input_id = _input_id(nodes)
    reachable = _reachable(input_id, edges)
    order = _topo(reachable, edges)

    warnings = [
        f"node '{nid}' (type '{nodes[nid]['type']}') is unreachable from input and was excluded"
        for nid in sorted(nodes, key=_idkey) if nid not in reachable
    ]
    terminals = sorted((nid for nid in reachable if not _has_succ(nid, edges)), key=_idkey)

    needs_add = any(nodes[nid]["type"] == "add" for nid in reachable)
    needs_pe = any(nodes[nid]["type"] == "positional_encoding" for nid in reachable)

    out = ["import torch", "import torch.nn as nn", "import math", "", "", _LAYER_ERROR_SRC]
    if needs_add:
        out += ["", "", _MERGE_ADD_SRC]
    if needs_pe:
        out += ["", "", _POS_ENC_SRC]
    out += ["", "", "class GeneratedModel(nn.Module):", "    def __init__(self):",
            "        super().__init__()"]

    module_lines = []
    for nid in order:
        ctor = constructor_code(nodes[nid])
        if ctor is not None:
            module_lines.append(f"        self.layer_{nid} = {ctor}")
    out += module_lines if module_lines else ["        pass"]

    out += ["", "    def forward(self, x):", "        outputs = {}"]
    for nid in order:
        node = nodes[nid]
        if node["type"] == "input":
            out.append(f'        outputs["{nid}"] = x')
            continue
        rhs = _forward_rhs(node, _preds(nid, edges, reachable))
        out += [
            "        try:",
            f'            outputs["{nid}"] = {rhs}',
            "        except Exception as e:",
            f'            raise LayerExecutionError("{nid}", "{node["type"]}", e)',
        ]

    if len(terminals) == 1:
        out.append(f'        return outputs["{terminals[0]}"]')
    else:
        items = ", ".join(f'outputs["{t}"]' for t in terminals)
        out.append(f"        return ({items})")
    out.append("")

    return "\n".join(out), terminals, warnings
