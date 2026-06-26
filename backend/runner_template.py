"""Fixed subprocess runner used by ``validator.py``.

Invoked as:  python runner_template.py <generated_module_path> <spec_json>

where ``spec_json`` is ``{"shape": [...], "dtype": "float32"|"int64", "bound": int}``.
It imports the generated module, builds the N=2 dummy batch, runs one forward
pass, and prints EXACTLY ONE JSON line to stdout:

  {"ok": true,  "output_shapes": [[...], ...]}
  {"ok": false, "node_id": ..., "layer_type": ..., "message": ...}

Runs in its own process so a malformed/hanging model can't take down the main
MCP/websocket process. All diagnostics go to stderr; stdout is the one JSON line.
"""

import importlib.util
import json
import sys

import torch


def _load_module(path):
    spec = importlib.util.spec_from_file_location("generated_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    module_path = sys.argv[1]
    spec = json.loads(sys.argv[2])
    shape, dtype, bound = spec["shape"], spec["dtype"], spec["bound"]

    mod = _load_module(module_path)

    if dtype == "int64":
        x = torch.randint(0, max(1, bound), (2, *shape), dtype=torch.long)
    else:
        x = torch.randn(2, *shape)

    try:
        model = mod.GeneratedModel()
        model.eval()
        with torch.no_grad():
            out = model(x)
    except mod.LayerExecutionError as e:
        print(json.dumps({"ok": False, "node_id": e.node_id,
                          "layer_type": e.layer_type, "message": str(e.original)}))
        return
    except Exception as e:  # construction error or non-layer failure
        print(json.dumps({"ok": False, "node_id": None, "layer_type": None,
                          "message": f"{type(e).__name__}: {e}"}))
        return

    if isinstance(out, (tuple, list)):
        shapes = [list(o.shape) for o in out]
    else:
        shapes = [list(out.shape)]
    print(json.dumps({"ok": True, "output_shapes": shapes}))


if __name__ == "__main__":
    main()
