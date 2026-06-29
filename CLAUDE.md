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
   Node `(x,y)` pixel positions are NEVER in the node schema (`{id,type,params}`), so the validator and
   codegen never see geometry. Two things layer on top of that, both kept *out* of node params:
   - The frontend owns pixel layout, recomputed every broadcast by a deterministic **edge-driven**
     layout (`frontend/src/layout.js` `computeLayout`): column = distance from the `input` node, so wiring
     `n1 -> n2` puts `n2` one column right of `n1` (and the `input` node is leftmost). Two overrides are
     layered on top, in order: a node the user **dragged** keeps its dragged position; else an agent
     **layout hint** `{col,row}` wins. Dragging snaps to the column/row grid.
   - The store keeps a **separate** optional `layout` map (`id -> {col,row}`) that an agent may set via
     `add_layer`/`add_layers`. It's broadcast inside `get_architecture` as a top-level `layout` key
     (alongside `nodes`/`edges`) purely as a placement *hint* for the canvas — it never reaches the
     validator or codegen (those read `_snapshot()`, which is nodes+edges only).

## Layer catalog scope (this build)

Beyond the spec's §8 set, the catalog is **broadly expanded** so all major NN families are buildable:
conv1d/conv3d/conv_transpose2d/upsample, adaptive_avg/max_pool2d, maxpool1d/avgpool1d,
sigmoid/tanh/leaky_relu/elu/softmax, batchnorm1d/groupnorm/instancenorm2d. Each catalog entry also
carries a `category` used for frontend color-coding. The implemented set is
input/conv/pooling/norm/activation/recurrent/attention/merge plus four buckets for layers that fit
none of those: `linear`, `shape` (flatten), `regularization` (dropout), `embedding`. Category is
frontend-only metadata, enforced by nothing. Adding a layer = one schema entry in `layers.py` + one
codegen mapping (`_CONSTRUCTORS`) in `codegen.py`.

## Agent tool surface (MCP)

12 tools, all thin adapters over one `ArchitectureStore` method (no graph logic in `mcp_tools.py`):
`add_layer`, `add_layers`, `update_layer`, `remove_layer`, `connect_layers`, `connect_layers_batch`,
`disconnect_layers`, `reset_architecture` (mutations, broadcast on success); `get_catalog`,
`get_architecture`, `validate_architecture`, `generate_code` (reads, never broadcast).

Built for an agent working **cold** (no repo access):
- **`get_catalog`** is the discovery surface — every type with its `category`, `required`, and
  `optional` params + defaults. Call it first instead of probing by trial-and-error.
- **Errors are written to be acted on**: an unknown type lists the known types; a missing-param error
  names *all* missing required params at once (not one per round-trip).
- **`add_layers` / `connect_layers_batch`** build a whole network in one **atomic** call each (all-or-
  nothing rollback), so a 100+ node model isn't hundreds of round-trips.
- **`add_layer`/`add_layers`** take an optional `layout` hint `{col,row}` (see invariant 8).

## Run commands

```bash
# Backend, full mode (MCP stdio + websocket on :8765) — this is what Claude Desktop launches
/home/hunain/base/bin/python backend/main.py

# Backend, frontend-dev mode (websocket only, no MCP stdio)
MODE=standalone /home/hunain/base/bin/python backend/main.py

# Frontend
cd frontend && npm install && npm run dev   # connects to ws://localhost:8765/ws
```

**Busy `:8765` (a stale/duplicate backend already running):** full mode does NOT
fight for the port. It keeps serving MCP (the agent's only contract) but runs
*without* the live canvas, and threads a `canvas_warning` into `build_mcp` so
**every mutation + `get_architecture` reply carries a "live canvas unavailable"
warning** — the agent's only channel is the MCP response (never stderr), so this
is how it learns the UI won't reflect its edits. A busy port means a *rival*
in-memory store owns the canvas; the fix is to stop the other backend (or set
`WS_PORT`), not to run two stores. (Separate processes never share the store.)

## Build order & status

Staged, gated build — see `.claude/handoff/` for each stage's scope, expected outcome, and the exact
verification gate. Do not start a stage until the previous gate passes.
0. ✅ Scaffolding/deps · 1. ✅ Core engine · 2. ✅ Store · 3. ✅ MCP tools · 4. ✅ Websocket · 5. Frontend · 6. Polish/README
