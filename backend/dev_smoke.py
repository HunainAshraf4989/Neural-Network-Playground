"""Scratch driver for Stage 1 - hand-built graphs, no MCP/ws/frontend.

Run:  /home/hunain/base/bin/python backend/dev_smoke.py

Builds one graph per NN family, validates each by real execution, and prints
the result. Then writes the CNN's generated code to /tmp/standalone_check.py and
imports it with only torch on the path to prove it is self-contained.
"""

import json
import subprocess
import sys

import codegen
import layers
import validator


def node(node_id, layer_type, **params):
    return {"id": node_id, "type": layer_type,
            "params": layers.validate_and_merge(layer_type, params)}


def state(nodes, edges):
    return {"nodes": nodes, "edges": [{"from": a, "to": b} for a, b in edges]}


# 1. CNN: conv -> pool -> flatten -> linear  (28x28 grayscale, 10 classes)
cnn = state(
    [node("n1", "input", shape=[1, 28, 28], dtype="float32"),
     node("n2", "conv2d", in_channels=1, out_channels=16, kernel_size=3),
     node("n3", "maxpool2d", kernel_size=2),
     node("n4", "flatten"),
     node("n5", "linear", in_features=16 * 13 * 13, out_features=10)],
    [("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5")],
)

# 2. Residual skip: input bypasses two convs via an add merge (channels must match)
residual = state(
    [node("n1", "input", shape=[16, 8, 8], dtype="float32"),
     node("n2", "conv2d", in_channels=16, out_channels=16, kernel_size=3, padding=1),
     node("n3", "conv2d", in_channels=16, out_channels=16, kernel_size=3, padding=1),
     node("n4", "add")],
    [("n1", "n2"), ("n2", "n3"), ("n1", "n4"), ("n3", "n4")],
)

# 3. Transformer encoder: embedding -> pos-enc -> encoder layer -> flatten -> linear
transformer = state(
    [node("n1", "input", shape=[16], dtype="int64"),
     node("n2", "embedding", num_embeddings=1000, embedding_dim=32),
     node("n3", "positional_encoding", d_model=32),
     node("n4", "transformer_encoder_layer", d_model=32, nhead=4),
     node("n5", "flatten"),
     node("n6", "linear", in_features=16 * 32, out_features=10)],
    [("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5"), ("n5", "n6")],
)

# 4. LSTM sequence model: input -> lstm  (output tensor propagates, hidden discarded)
lstm = state(
    [node("n1", "input", shape=[16, 8], dtype="float32"),
     node("n2", "lstm", input_size=8, hidden_size=16)],
    [("n1", "n2")],
)

# 5. Conv-transpose autoencoder: conv/pool encoder -> conv_transpose/upsample decoder -> sigmoid
autoencoder = state(
    [node("n1", "input", shape=[1, 16, 16], dtype="float32"),
     node("n2", "conv2d", in_channels=1, out_channels=8, kernel_size=3, padding=1),
     node("n3", "maxpool2d", kernel_size=2),
     node("n4", "conv2d", in_channels=8, out_channels=4, kernel_size=3, padding=1),
     node("n5", "maxpool2d", kernel_size=2),
     node("n6", "conv_transpose2d", in_channels=4, out_channels=8, kernel_size=2, stride=2),
     node("n7", "upsample", scale_factor=2),
     node("n8", "conv2d", in_channels=8, out_channels=1, kernel_size=3, padding=1),
     node("n9", "sigmoid")],
    [("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5"),
     ("n5", "n6"), ("n6", "n7"), ("n7", "n8"), ("n8", "n9")],
)

# Broken: second conv expects 8 in-channels but receives 16 -> fails at n3
broken = state(
    [node("n1", "input", shape=[1, 28, 28], dtype="float32"),
     node("n2", "conv2d", in_channels=1, out_channels=16, kernel_size=3),
     node("n3", "conv2d", in_channels=8, out_channels=32, kernel_size=3)],
    [("n1", "n2"), ("n2", "n3")],
)

FAMILIES = [
    ("CNN", cnn),
    ("Residual skip", residual),
    ("Transformer encoder", transformer),
    ("LSTM sequence", lstm),
    ("Conv-transpose autoencoder", autoencoder),
]


def main():
    all_ok = True
    for name, graph in FAMILIES:
        res = validator.validate(graph)
        ok = res.get("valid") is True
        all_ok &= ok
        print(f"[{name}] valid: {res.get('valid')}  "
              f"output_shapes: {res.get('output_shapes')}  "
              f"output_node_ids: {res.get('output_node_ids')}")
        if res.get("warnings"):
            print(f"    warnings: {res['warnings']}")
        if not ok:
            print(f"    error: {res.get('error')}")

    print("\n--- broken case (expect valid: false naming n3 / conv2d) ---")
    res = validator.validate(broken)
    print(f"[Broken channel mismatch] {json.dumps(res)}")
    broken_ok = (res.get("valid") is False
                 and res.get("error", {}).get("node_id") == "n3"
                 and res.get("error", {}).get("layer_type") == "conv2d")
    print(f"    correctly identified broken node: {broken_ok}")

    # Standalone self-containment check: run the CNN's generated code with only torch.
    print("\n--- standalone self-containment check ---")
    code, _, _ = codegen.generate(cnn)
    with open("/tmp/standalone_check.py", "w") as f:
        f.write(code)
    proc = subprocess.run([sys.executable, "/tmp/standalone_check.py"],
                          capture_output=True, text=True, cwd="/tmp")
    standalone_ok = proc.returncode == 0
    print(f"    /tmp/standalone_check.py imports/executes cleanly: {standalone_ok}")
    if not standalone_ok:
        print(f"    stderr: {proc.stderr.strip()}")

    print(f"\nALL FAMILIES VALID: {all_ok} | BROKEN DETECTED: {broken_ok} | "
          f"STANDALONE: {standalone_ok}")


if __name__ == "__main__":
    main()
