"""Architecture validation by REAL execution in an isolated subprocess.

This is the only place shape-correctness is decided. It generates code, writes
it to a per-invocation temp file (deleted when the run ends — concurrent
validations never share a path), then runs ``runner_template.py`` against it
under a timeout (``NN_VALIDATION_TIMEOUT_S``, default 10 s). Errors come back
as structured ``LayerExecutionError`` fields (node_id/layer_type/message),
never parsed from a generic traceback.
"""

import json
import os
import subprocess
import sys
import tempfile

import codegen

RUNNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner_template.py")
TIMEOUT_SECONDS = float(os.environ.get("NN_VALIDATION_TIMEOUT_S", "10"))
# Pre-flight resource guard: a graph estimated above this many trainable params is
# rejected *before* we spawn torch, so a pathological model (e.g. a linear layer
# with 100k in/out = 10B params) gets a clean message instead of thrashing the
# machine or aborting OpenBLAS. This is a memory guard, NOT a shape check
# (invariant 4 — shape correctness is still decided only by real execution below).
# The subprocess also caps its own RAM (runner_template._cap_memory) as a hard net.
MAX_PARAMS = int(os.environ.get("NN_VALIDATION_MAX_PARAMS", str(500_000_000)))


def _kernel_volume(k):
    if isinstance(k, (list, tuple)):
        vol = 1
        for v in k:
            vol *= v
        return vol
    return k


def _node_param_estimate(node):
    """Rough trainable-weight count for one layer from its params alone (weights
    dominate; biases/norms ignored). Pre-flight resource estimate only."""
    t = node["type"]
    p = node.get("params", {})
    try:
        if t == "linear":
            return p["in_features"] * p["out_features"]
        if t in ("conv1d", "conv2d", "conv3d", "conv_transpose2d"):
            return p["in_channels"] * p["out_channels"] * _kernel_volume(p["kernel_size"])
        if t == "embedding":
            return p["num_embeddings"] * p["embedding_dim"]
        if t in ("rnn", "lstm", "gru"):
            gates = {"rnn": 1, "lstm": 4, "gru": 3}[t]
            h, i = p["hidden_size"], p["input_size"]
            dirs = 2 if p.get("bidirectional") else 1
            return gates * dirs * p.get("num_layers", 1) * h * (i + h)
        if t == "multihead_attention":
            return 4 * p["embed_dim"] * p["embed_dim"]
        if t == "transformer_encoder_layer":
            d = p["d_model"]
            return 4 * d * d + 2 * d * p.get("dim_feedforward", 2048)
    except (KeyError, TypeError):
        return 0
    return 0


def _estimate_params(state):
    """Total estimated params and the single heaviest node (id, count)."""
    total, worst_id, worst = 0, None, 0
    for n in state["nodes"]:
        c = _node_param_estimate(n)
        total += c
        if c > worst:
            worst_id, worst = n["id"], c
    return total, worst_id, worst


def _embedding_bound(state, input_id):
    """BFS from input; bound int64 indices by the first downstream embedding's
    ``num_embeddings`` so random indices don't trigger spurious lookup errors.
    """
    nodes = {n["id"]: n for n in state["nodes"]}
    adj = {}
    for e in state["edges"]:
        adj.setdefault(e["from"], []).append(e["to"])
    seen, queue = {input_id}, [input_id]
    while queue:
        nid = queue.pop(0)
        if nodes[nid]["type"] == "embedding":
            return nodes[nid]["params"]["num_embeddings"]
        for m in adj.get(nid, []):
            if m not in seen:
                seen.add(m)
                queue.append(m)
    return 1000


def validate(state):
    """Validate ``state`` and return the ``validate_architecture`` tool response."""
    input_node = next((n for n in state["nodes"] if n["type"] == "input"), None)
    if input_node is None:
        return {"valid": False, "error": {"message": "no input node defined"}, "warnings": []}

    total_params, worst_id, worst = _estimate_params(state)
    if total_params > MAX_PARAMS:
        return {
            "valid": False,
            "error": {
                "node_id": worst_id,
                "message": (
                    f"architecture too large to validate safely: ~{total_params:,} parameters"
                    + (f" (largest at {worst_id}: ~{worst:,})" if worst_id else "")
                    + f" exceeds the {MAX_PARAMS:,}-parameter limit. Executing it could exhaust "
                    "memory; shrink the layers, or raise NN_VALIDATION_MAX_PARAMS "
                    "(and NN_VALIDATION_MEM_MB) if your machine can take it."
                ),
            },
            "warnings": [],
        }

    code, terminals, warnings = codegen.generate(state)
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="nn_architect_generated_",
            delete=False) as f:
        f.write(code)
        generated_path = f.name

    spec = json.dumps({
        "shape": input_node["params"]["shape"],
        "dtype": input_node["params"]["dtype"],
        "bound": _embedding_bound(state, input_node["id"]),
    })

    try:
        proc = subprocess.run(
            [sys.executable, RUNNER_PATH, generated_path, spec],
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {"valid": False,
                "error": {"message": f"validation timed out after {TIMEOUT_SECONDS}s"},
                "warnings": warnings}
    finally:
        try:
            os.unlink(generated_path)
        except OSError:
            pass

    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if not line:
        return {"valid": False,
                "error": {"message": "validation runner produced no output; "
                                     f"stderr: {proc.stderr.strip()[:500]}"},
                "warnings": warnings}
    try:
        res = json.loads(line)
    except json.JSONDecodeError:
        return {"valid": False,
                "error": {"message": f"could not parse runner output: {line[:300]}"},
                "warnings": warnings}

    if res.get("ok"):
        return {"valid": True, "output_shapes": res["output_shapes"],
                "output_node_ids": terminals, "warnings": warnings}

    if res.get("node_id"):
        error = {"node_id": res["node_id"], "layer_type": res["layer_type"],
                 "message": res.get("message")}
    else:
        error = {"message": res.get("message")}
    return {"valid": False, "error": error, "warnings": warnings}
