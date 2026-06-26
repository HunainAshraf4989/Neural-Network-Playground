"""Architecture validation by REAL execution in an isolated subprocess.

This is the only place shape-correctness is decided. It generates code, writes
it to a fixed temp path, then runs ``runner_template.py`` against it under a
10s timeout. Errors come back as structured ``LayerExecutionError`` fields
(node_id/layer_type/message), never parsed out of a generic traceback.
"""

import json
import os
import subprocess
import sys

import codegen

GENERATED_PATH = "/tmp/_nn_architect_generated.py"
RUNNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner_template.py")
TIMEOUT_SECONDS = 10


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

    code, terminals, warnings = codegen.generate(state)
    with open(GENERATED_PATH, "w") as f:
        f.write(code)

    spec = json.dumps({
        "shape": input_node["params"]["shape"],
        "dtype": input_node["params"]["dtype"],
        "bound": _embedding_bound(state, input_node["id"]),
    })

    try:
        proc = subprocess.run(
            [sys.executable, RUNNER_PATH, GENERATED_PATH, spec],
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {"valid": False,
                "error": {"message": f"validation timed out after {TIMEOUT_SECONDS}s"},
                "warnings": warnings}

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
