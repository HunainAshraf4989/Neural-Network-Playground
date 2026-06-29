# CLAUDE.md — nn-architect-mcp

Guidance for any agent (or human) working in this repo. Read this before touching code.

## What this is

An **MCP server for collaborative, agent-driven neural-network architecture design with live shape
validation**. An LLM agent builds a network graph via MCP tool calls; a human watches it render live
on a React canvas and can edit the same graph; either side hands control back to the other. The
validator actually *executes* generated PyTorch against a dummy tensor and returns structured,
per-node shape errors — not raw tracebacks.

This is a **portfolio prototype**, not a product. Lead all marketing/README copy with
"collaborative agent-driven design + live validation," not "neural net builder."

Canonical spec: `Tentative plan.md` (sections referenced as §N below).
Approved build plan with staged gates: `~/.claude/plans/iridescent-bubbling-noodle.md`.
Per-stage handoff docs: `.claude/handoff/STAGE_*.md`.
always use /karpathy-guidelines skill.

## Environment — IMPORTANT

- **All Python deps live in the venv at `/home/hunain/base`. Always use it.** No project-local venv,
  no Docker.
  - Run Python: `/home/hunain/base/bin/python ...`
  - Install: `/home/hunain/base/bin/pip install ...`
- Python 3.12, Node 20, npm 10 on host.

## Non-negotiable invariants (break these and the system silently corrupts)

1. **stdout is reserved for MCP protocol framing.** Any stray `print()` or logging to stdout
   corrupts the MCP stream and silently kills the connection. **All logs/debug go to stderr.**
2. **One `ArchitectureStore` is the only mutation path.** MCP tool handlers (`mcp_tools.py`) and the
   websocket handler (`ws_app.py`) are *thin adapters* over the same store methods. Never write a
   second code path that mutates the graph — they will drift and the canvas will desync.
3. **Generated code must be fully self-contained.** It imports only `torch`/`math`, and inlines
   `LayerExecutionError` and (when needed) `PositionalEncoding`. Acceptance test: copy
   `generate_code` output to a fresh `.py` outside the repo, run with only `torch` installed.
4. **Shape correctness is checked ONLY by real execution** inside `validate_architecture` — never by
   static analysis in `add_layer`/`connect_layers`. The catalog's "shape effect" column is
   documentation, not enforced logic. Don't build a second static shape-checker.
5. **Tensor conventions:** `batch_first=True` everywhere (never an exposed param); image tensors
   `(N,C,H,W)`, sequence tensors `(N,L,E)`. Validation dummy batch is fixed **N=2**.
6. **Self-attention only:** `multihead_attention` forward is `self.layer(x, x, x)`. No cross-attention.
7. **Validation runs in a subprocess** (`subprocess.run(..., timeout=10)`) so a bad model can't take
   down the main process. Errors come back as structured `LayerExecutionError`
   (`node_id`/`layer_type`/`original`), never parsed from generic strings.
8. **IDs** are an incremental counter (`n1`, `n2`, …), reset to 0 on `reset_architecture`. No UUIDs.
   Node `(x,y)` positions are NOT in the shared schema — layout is frontend-only, recomputed each update.

## Layer catalog scope (this build)

Beyond the spec's §8 set, the catalog is **broadly expanded** so all major NN families are buildable:
conv1d/conv3d/conv_transpose2d/upsample, adaptive_avg/max_pool2d, maxpool1d/avgpool1d,
sigmoid/tanh/leaky_relu/elu/softmax, batchnorm1d/groupnorm/instancenorm2d. Each catalog entry also
carries a `category` used for frontend color-coding. The implemented set is
input/conv/pooling/norm/activation/recurrent/attention/merge plus four buckets for layers that fit
none of those: `linear`, `shape` (flatten), `regularization` (dropout), `embedding`. Category is
frontend-only metadata, enforced by nothing. Adding a layer = one schema entry in `layers.py` + one
codegen mapping (`_CONSTRUCTORS`) in `codegen.py`.

## Run commands

```bash
# Backend, full mode (MCP stdio + websocket on :8765) — this is what Claude Desktop launches
/home/hunain/base/bin/python backend/main.py

# Backend, frontend-dev mode (websocket only, no MCP stdio)
MODE=standalone /home/hunain/base/bin/python backend/main.py

# Frontend
cd frontend && npm install && npm run dev   # connects to ws://localhost:8765/ws
```

## Build order & status

Staged, gated build — see `.claude/handoff/` for each stage's scope, expected outcome, and the exact
verification gate. Do not start a stage until the previous gate passes.
0. ✅ Scaffolding/deps · 1. ✅ Core engine · 2. ✅ Store · 3. ✅ MCP tools · 4. ✅ Websocket · 5. Frontend · 6. Polish/README
