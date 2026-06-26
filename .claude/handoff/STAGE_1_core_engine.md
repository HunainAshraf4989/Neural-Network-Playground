# Stage 1 — Core engine (layers, codegen, validator)

## Scope
The pure-Python heart: the layer catalog, code generation, and subprocess validation. **No MCP, no
websocket, no frontend.** Driven by hand-written `AppState` dicts via a scratch `dev_smoke.py`.

## Files
- `backend/layers.py` — catalog. Each entry: `category`, `required` params, `optional` params with
  defaults, and a codegen hook (how to emit the `nn.Module` ctor + the forward call). Includes the
  broadly-expanded set (see CLAUDE.md "Layer catalog scope"). Validation here is **structural only**
  (unknown type, missing required param) — never tensor-shape math.
- `backend/codegen.py` — Kahn topo-sort of the subgraph reachable from `input`; emit a self-contained
  module: imports (`torch`, `torch.nn as nn`, `math`), inline `class LayerExecutionError`, inline
  `class PositionalEncoding` if needed, then the model class. `__init__` instantiates
  `self.layer_<id>` for non-merge nodes; `forward(self, x)` walks topo order with an `outputs` dict,
  wraps each node in `try/except Exception as e: raise LayerExecutionError(id, type, e)`, handles
  `add`/`concat` merges (concat in edge-insertion order), returns terminal tensor(s) per §8. Orphan
  nodes excluded → returned as warnings.
- `backend/runner_template.py` — fixed runner script: imports the generated module, builds the model,
  runs the dummy forward, prints **one JSON line** to stdout: `{"ok": true, "output_shapes": [...]}`
  or `{"ok": false, "node_id", "layer_type", "message"}`.
- `backend/validator.py` — write codegen output to a fixed temp path; build dummy input
  (`torch.randn(2, *shape)`; int64 bounded by first downstream `embedding.num_embeddings`, default
  1000); `subprocess.run(..., timeout=10)`; parse the JSON line into the tool-response shape; timeout
  → `{"valid": false, "error": {"message": "validation timed out after 10s"}}`.
- `backend/dev_smoke.py` (scratch, gitignored area ok) — hand-built graphs for each family.

## Invariants to honor
stderr-only logging · self-contained generated code · correctness only via execution · N=2 ·
`batch_first=True` · self-attention `self.layer(x,x,x)` · structured `LayerExecutionError`.

## Expected outcome
Five family models generate code that runs standalone and validates:
1. CNN (conv2d → pool → flatten → linear)
2. Residual skip (two convs bypassed by an `add` merge)
3. Transformer-encoder (embedding → positional_encoding → transformer_encoder_layer → flatten/linear)
4. LSTM sequence model
5. **Conv-transpose autoencoder** (conv2d/pool encoder → conv_transpose2d/upsample decoder → sigmoid)

One intentionally channel-mismatched graph returns `valid: false` naming the correct `node_id`/`layer_type`.

## Verification gate
```bash
/home/hunain/base/bin/python backend/dev_smoke.py
```
Prints `valid: true` + output shapes for all five families; the broken case reports the right node.
Then copy one generated model to `/tmp/standalone_check.py` and run it with only torch — it runs
unmodified.

## Status
⬜ not started.
