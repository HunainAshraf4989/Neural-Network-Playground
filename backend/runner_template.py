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
import os
import sys


def _cap_memory():
    """Cap THIS validation process's memory before torch loads, so a pathological
    architecture (e.g. a linear layer with 100k features → tens of GB of weights)
    fails with a clean MemoryError instead of driving the whole machine into
    swap-thrash or tripping the OS OOM killer - which may kill unrelated processes.
    The 10s timeout in ``validator.py`` guards against *hangs*; this guards against
    *memory*. The cap is on address space (RLIMIT_AS, so it also covers mmap'd
    tensor allocations) and is tunable via ``NN_VALIDATION_MEM_MB``.

    The default is 8192 MB, not a tighter-looking number, because RLIMIT_AS counts
    *reserved address space*, not resident memory, and torch reserves a lot of it up
    front: measured here, ``import torch`` alone takes ~3.4 GB of VA (one allocator
    arena per thread) and a trivial 256-dim encoder forward reaches ~4.2 GB. A 4096
    cap therefore sat BELOW torch's own floor and failed legitimate models with a
    confusing allocator error before their size ever mattered. 8192 clears the floor
    with headroom while still killing genuinely pathological graphs. The real defence
    against absurd models is the param-count pre-flight in ``validator.py``
    (``NN_VALIDATION_MAX_PARAMS``); this is the backstop for what that misses, mainly
    big conv activation maps.

    ``resource`` is POSIX-only, so this is a no-op elsewhere; the subprocess
    isolation still stands either way.
    """
    try:
        import resource
    except ImportError:
        return
    target = int(os.environ.get("NN_VALIDATION_MEM_MB", "8192")) * 1024 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        target = min(target, hard)  # can't raise the hard limit, only lower the soft
    resource.setrlimit(resource.RLIMIT_AS, (target, hard))


_cap_memory()

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
