# nn-architect-mcp (working title — rename freely)

An MCP server for **collaborative, agent-driven neural network architecture design with live shape validation**. An LLM agent (Claude Desktop, Claude Code, Cowork, or any MCP client) builds a network by calling tools; a human watches it appear live on a canvas in the browser and can edit the same graph; either side can hand control back to the other.

This document is a complete build spec. It is written so that an agent building this from scratch needs to make **zero unstated design decisions**. Every ambiguous point that came up while scoping this has been pinned down explicitly — see "Decisions Already Made For You" near the end if you only read one section.

---

## 1. Why this exists / framing for the build

The "drag-and-drop neural network builder that exports PyTorch code" category is already crowded (several near-identical public projects exist). The differentiator for this project is **not** the builder UI — it's that the architecture graph is exposed as a live, shared, agent-editable surface over MCP, with a validation tool that actually executes the generated code against a dummy tensor and returns structured, per-node shape errors. Keep that framing in the README's own marketing copy, demo script, and any future polish: lead with "collaborative agent-driven design + live validation," not "neural net builder, now with MCP."

## 2. Goals

- An LLM agent connected via MCP can construct an arbitrary directed-acyclic neural network graph — including CNN blocks, RNN/LSTM/GRU blocks, Transformer blocks, and residual/skip-connection patterns (add or concat merges) — purely through tool calls.
- A human watching the web frontend sees every change appear live, with no manual refresh.
- A human can also add, edit, remove, or rewire nodes directly in the UI, and the agent sees those changes the next time it calls `get_architecture`.
- The agent can ask the system to validate the current graph and get back a structured, node-specific error if shapes don't line up — not a raw Python traceback.
- The agent can request final, standalone, runnable PyTorch source code for the current graph at any time.

## 3. Out of scope for this PoC (explicitly excluded — nothing added beyond this list)

- In-app/in-browser model **training**. No optimizer, no loss tracking, no datasets. This project designs architectures; it does not train them.
- **Multi-framework export.** PyTorch only. No Keras/TensorFlow/JAX code generation.
- **Remote hosting, multi-user auth, or OAuth.** This runs locally only — a local MCP stdio server plus a local dev server. (Listed under Future Work below, not built now.)
- **Visual polish / custom theming.** Default styling from whatever component library is used is acceptable. This is a functionality PoC, not a design showcase.

## 4. High-level architecture

One Python process, one asyncio event loop, two things running concurrently inside it:

1. **MCP server** (stdio transport) — the tool surface an LLM agent calls.
2. **FastAPI app** (websocket + a couple of REST endpoints) — serves the frontend's live state sync.

Both sit on top of **one shared in-memory store** (`ArchitectureStore`). This is the single source of truth. MCP tool handlers and the websocket message handler are both thin adapters that call the *same* store methods — there is never a second, parallel code path for mutating the graph. This matters: if you implement "apply a human edit from the UI" as separate logic from "apply an agent tool call," they will drift and the canvas will eventually show something the agent doesn't think exists.

```
┌─────────────────────┐        stdio (stdin/stdout)        ┌──────────────────────────┐
│  MCP client          │ ◄─────────────────────────────────► │  MCP tool handlers       │
│  (Claude Desktop,    │                                     │  (mcp_tools.py)          │
│   Claude Code, etc.) │                                     └────────────┬─────────────┘
└─────────────────────┘                                                  │
                                                                          ▼
                                                              ┌──────────────────────────┐
                                                              │   ArchitectureStore       │
                                                              │   (store.py)              │
                                                              │   single source of truth  │
                                                              └────────────┬─────────────┘
                                                                          ▲
┌─────────────────────┐      websocket (ws://localhost:8765/ws)         │
│  React frontend      │ ◄─────────────────────────────────────────────►│
│  (canvas, palette,   │                                     ┌──────────────────────────┐
│   params panel)      │                                     │  FastAPI ws handler       │
└─────────────────────┘                                     │  (ws_app.py)              │
                                                              └──────────────────────────┘
```

**Critical gotcha:** the MCP stdio transport uses the process's actual stdout as the protocol channel. Any stray `print()` or unstructured logging to stdout will corrupt the MCP message stream and silently break the connection. All logging/debug output must go to **stderr**, never stdout, anywhere in this process.

## 5. Tech stack

- **Backend:** Python 3.11+, `fastapi`, `uvicorn[standard]`, `websockets`, `torch` (CPU build is fine), `mcp` (official Anthropic Python SDK — use the `FastMCP` high-level class from `mcp.server.fastmcp` for tool registration via decorators, not the low-level protocol classes), `pydantic`.
- **Frontend:** React + Vite, plain JavaScript (not TypeScript — keep setup friction low), and a node-graph/diagramming library with support for arbitrary directed graphs and multiple edges into one node (React Flow is the standard choice for this; confirm the current package name on npm at install time since it has been renamed before — check `https://www.npmjs.com` directly rather than assuming).
- No database. No persistence. State lives in memory and is lost on process restart — this is intentional for a PoC, not an oversight (see Future Work).

## 6. Repository structure

```
nn-architect-mcp/
├── README.md
├── backend/
│   ├── requirements.txt
│   ├── main.py                  # entrypoint — starts MCP stdio loop + uvicorn concurrently
│   ├── store.py                 # ArchitectureStore: all mutation methods, the one source of truth
│   ├── layers.py                # layer catalog: param schemas, defaults, per-type validation
│   ├── codegen.py                # topological sort + code-string assembly
│   ├── validator.py              # builds dummy input, runs generated code in a subprocess
│   ├── runner_template.py        # fixed subprocess runner script used by validator.py
│   ├── mcp_tools.py              # FastMCP tool registrations, each calling into store.py
│   └── ws_app.py                 # FastAPI app + websocket handler, calling into store.py
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx               # websocket connection + state
│       ├── LayerNode.jsx         # generic custom node component, parameterized by type
│       ├── LayerPalette.jsx      # draggable list of catalog layer types
│       ├── ParamsPanel.jsx       # side panel for editing a selected node's params
│       └── layout.js              # manual layered auto-layout (see §11)
└── claude_desktop_config.snippet.json
```

## 7. Data model

```python
# A node in the graph
Node = {
    "id": str,           # e.g. "n1", "n2", ... — see ID scheme below
    "type": str,         # one of the catalog type names in §8, including "input"
    "params": dict,       # type-specific, see §8
}

# A directed edge
Edge = {
    "from": str,  # source node id
    "to": str,    # target node id
}

# The whole shared state
AppState = {
    "nodes": list[Node],
    "edges": list[Edge],   # order matters — see "concat input order" decision below
}
```

**ID scheme:** incremental string IDs (`"n1"`, `"n2"`, ...) from a counter on the store, not UUIDs. Readable in logs and demos. Counter resets to 0 on `reset_architecture`.

**Positions are not part of this schema.** No `x`/`y` on a node. Layout is purely a frontend concern, recomputed on every state update — see §11.

## 8. Layer catalog

Tensor conventions used throughout: image-like tensors are `(N, C, H, W)`; sequence-like tensors are `(N, L, E)` — batch, sequence length, feature/embedding size. `batch_first=True` is hardcoded everywhere a PyTorch module exposes that flag; it is never an exposed param, to keep one tensor convention system-wide.

**Shape correctness is determined exclusively by actually running the generated forward pass inside `validate_architecture` (§10) — never by static analysis inside `add_layer` or `connect_layers`.** Those two tools only ever check structural validity (unknown type, missing required param, cycle, etc.), never tensor-shape compatibility. Do not build a second, separate static shape-checker — one source of truth for correctness, the real execution.

| type | category | required params | optional params (default) | shape effect |
|---|---|---|---|---|
| `input` | input | `shape` (list[int], excludes batch dim), `dtype` ("float32" or "int64") | — | output = `shape` |
| `conv2d` | layer | `in_channels` (int), `out_channels` (int), `kernel_size` (int or [int,int]) | `stride` (1), `padding` (0), `dilation` (1) | `H_out = floor((H + 2·pad − dilation·(k−1) − 1)/stride + 1)`, same for W; channels = `out_channels` |
| `maxpool2d` | layer | `kernel_size` | `stride` (= kernel_size), `padding` (0) | same formula as conv2d, `ceil_mode=False` fixed; channels unchanged |
| `avgpool2d` | layer | `kernel_size` | `stride` (= kernel_size), `padding` (0) | same as maxpool2d |
| `batchnorm2d` | layer | `num_features` (int) | — | unchanged |
| `relu` | layer | — | — | unchanged |
| `gelu` | layer | — | — | unchanged |
| `flatten` | layer | — | `start_dim` (1) | `(N, prod(dims from start_dim onward))` |
| `linear` | layer | `in_features`, `out_features` | — | last dim → `out_features` |
| `dropout` | layer | — | `p` (0.5) | unchanged |
| `rnn` / `lstm` / `gru` | layer | `input_size`, `hidden_size` | `num_layers` (1), `bidirectional` (false) | `(N, L, hidden_size × (2 if bidirectional else 1))`. Only the `output` tensor propagates downstream — hidden/cell state is discarded. |
| `embedding` | layer | `num_embeddings`, `embedding_dim` | — | input must be int64 indices `(N, L)` → output `(N, L, embedding_dim)` |
| `positional_encoding` | layer | `d_model` | `max_len` (5000) | unchanged — fixed sinusoidal encoding, no learnable params, added elementwise |
| `layernorm` | layer | `normalized_shape` (int or list) | — | unchanged |
| `multihead_attention` | layer | `embed_dim`, `num_heads` | `dropout` (0.0) | unchanged — **self-attention only**: forward call is `self.layer(x, x, x)` (query=key=value=input). No cross-attention/encoder-decoder support. |
| `transformer_encoder_layer` | layer | `d_model`, `nhead` | `dim_feedforward` (2048), `dropout` (0.1) | unchanged |
| `add` | merge | — | — | elementwise sum of **all** incoming tensors. Requires ≥2 incoming edges. Shapes must be exactly equal — assert this explicitly in generated code rather than relying on broadcasting, so a broadcastable-but-wrong case still gets flagged. |
| `concat` | merge | `dim` (int, no default — must be explicit) | — | `torch.cat([...], dim=params.dim)`. Requires ≥2 incoming edges. **Input order = the order `connect_layers` was called in, not node-id order.** |

Graph rules that apply regardless of type:
- Exactly one `input` node per graph. `input` nodes can never have an incoming edge.
- Every non-`input`, non-merge node requires exactly one incoming edge.
- `add`/`concat` require two or more.
- The graph must stay a DAG — `connect_layers` runs a cycle check (simple DFS from the proposed target back to the proposed source) and rejects any edge that would create a cycle. Self-loops (`from_id == to_id`) are rejected immediately, same call.
- A node with no outgoing edges is a **terminal node**. If exactly one exists, `generate_code`'s forward returns that single tensor. If more than one exists, it returns a tuple of all of them, ordered by ascending node id.
- A node unreachable from the `input` node (orphaned mid-edit) is not an error. It's excluded from codegen/validation and surfaced as a non-fatal warning string, so an agent mid-design isn't blocked by a scratch node it hasn't wired up yet.

## 9. MCP tool surface (9 tools)

All tool handlers call into `ArchitectureStore` methods — never duplicate this logic in `ws_app.py`.

**`add_layer(type: str, params: dict) -> {node_id: str}`**
Validates `type` against the catalog and `params` against that type's required/optional fields (applying defaults for anything omitted). Rejects a second `input` node — error message tells the agent to `remove_layer` the existing one first if it wants to redefine the input.

**`update_layer(node_id: str, params: dict) -> {node_id, params}`**
Merges given params into the existing ones (partial update allowed) and re-validates the merged result against the type's schema.

**`remove_layer(node_id: str) -> {removed: true, edges_removed: [...]}`**
Cascade-deletes any edges touching this node.

**`connect_layers(from_id: str, to_id: str) -> {edge: {from, to}}`**
Errors: either id missing, self-loop, would create a cycle, `to_id` is an `input` node, or this exact edge already exists (reject as duplicate rather than silently no-op-ing, so the agent always knows whether its call changed anything).

**`disconnect_layers(from_id: str, to_id: str) -> {removed: true}`**
Errors if the edge doesn't exist.

**`get_architecture() -> {nodes: [...], edges: [...]}`**
Full current state, edges in stored (insertion) order.

**`validate_architecture() -> {valid: bool, ...}`**
No input params — always validates against the single `input` node's own declared shape/dtype, with a fixed dummy batch size of **N=2** (chosen specifically so batch-dimension bugs aren't masked the way they would be at N=1). See §10 for mechanism. On success: `{"valid": true, "output_shapes": [...], "output_node_ids": [...], "warnings": [...]}`. On failure: `{"valid": false, "error": {"node_id", "layer_type", "message"}, "warnings": [...]}`.

**`generate_code() -> {code: str}`**
Always emits best-effort code from the current graph — does not require a prior successful `validate_architecture` call. The agent is responsible for validating first if it wants a correctness guarantee.

**`reset_architecture() -> {reset: true}`**
Clears all nodes/edges and resets the ID counter to 0.

## 10. Code generation and validation mechanism

**Generation (`codegen.py`):**
1. Topologically sort the subgraph reachable from `input` (Kahn's algorithm). Orphan nodes are excluded (see §8) and listed in `warnings`.
2. Emit a self-contained Python file: imports (`torch`, `torch.nn as nn`, `math`), an inline `class LayerExecutionError(Exception)` (fields: `node_id`, `layer_type`, `original`), an inline `class PositionalEncoding(nn.Module)` if any `positional_encoding` node is present, then the model class.
3. `__init__`: for each non-merge node, instantiate `self.layer_<node_id> = <corresponding nn.Module>(...)`. Merge nodes (`add`/`concat`) have no instantiated module — they're forward-only operations.
4. `forward(self, x)`: maintain a dict `outputs = {"<input_node_id>": x}`. Walk the topo order; for each node, wrap its computation in `try/except Exception as e: raise LayerExecutionError(node_id, layer_type, e)`. Single-input nodes call `self.layer_<id>(outputs[<predecessor_id>])`. Merge nodes gather all predecessor outputs (concat: in edge-insertion order) and apply `torch.add`/`torch.cat`.
5. Return the terminal tensor(s) per the rule in §8.

**Why `LayerExecutionError` and not string-parsing a traceback:** it carries structured fields directly, so the validator never has to guess at error-message formats to figure out which node failed.

**This generated file must be fully self-contained** — no imports from this project's own backend package, only from `torch`/`math`. This is what makes the acceptance test "copy generate_code's output into a standalone .py file and run it with nothing but torch installed" actually work.

**Validation (`validator.py`):**
1. Get code from `codegen.py`, write it to a fixed temp path (e.g. `/tmp/_nn_architect_generated.py`), overwritten each call.
2. Build the dummy input tensor from the `input` node: `torch.randn(2, *shape)` for `dtype="float32"`. For `dtype="int64"`, bound the random indices by the `num_embeddings` of the first downstream `embedding` node found via BFS from input (default bound 1000 if none found) — otherwise random int64 indices would routinely trigger spurious "index out of range" failures that have nothing to do with the architecture itself.
3. Run a fixed runner script in a subprocess (`subprocess.run(..., timeout=10)`) that imports the generated module, builds the model, runs the dummy forward pass, and prints one JSON line to stdout: `{"ok": true, "output_shapes": [...]}` or, on catching `LayerExecutionError`, `{"ok": false, "node_id": ..., "layer_type": ..., "message": ...}`.
4. Parse that JSON line in the parent process and shape it into the `validate_architecture` tool response. A timeout becomes `{"valid": false, "error": {"message": "validation timed out after 10s"}}`.
5. Subprocess isolation is intentional — a malformed generated model crashing or hanging should never take down the main MCP/websocket process.

## 11. Backend process architecture (`main.py`)

Single process, single asyncio event loop, running concurrently via `asyncio.gather`:
- The `FastMCP` stdio server loop (registers the 9 tools from `mcp_tools.py`).
- `uvicorn.Server(...).serve()` for the FastAPI app in `ws_app.py`, on a fixed local port (use `8765`).

Both share one `ArchitectureStore` instance, constructed once in `main.py` and passed by reference into both `mcp_tools.py` and `ws_app.py`. Guard mutations in the store with an `asyncio.Lock` — not strictly required for correctness on a single event loop, but cheap and removes any need to reason about interleaving later.

**Run modes**, controlled by an env var or CLI flag (e.g. `MODE=standalone`):
- Default mode: runs both the MCP stdio loop and the FastAPI server. This is the mode Claude Desktop will launch when you register it as an MCP server — Desktop owns the process lifecycle (starts it on connect, kills it on disconnect).
- `standalone` mode: skips the MCP stdio loop, runs only the FastAPI/websocket server. Use this for frontend development so you're not dependent on having Claude Desktop running just to see the canvas.

## 12. WebSocket protocol (frontend ⟷ backend)

This intentionally mirrors the MCP tool surface 1:1 — the websocket handler is a thin adapter over the exact same `ArchitectureStore` methods the MCP tools call, never separate logic.

**Server → client**, sent on connect and after every successful mutation, broadcast to all connected clients:
```json
{"type": "state", "data": {"nodes": [...], "edges": [...]}}
```

**Client → server** (human edits), each one mapped directly to a store method:
```json
{"type": "add_layer", "layer_type": "...", "params": {...}}
{"type": "update_layer", "node_id": "...", "params": {...}}
{"type": "remove_layer", "node_id": "..."}
{"type": "connect_layers", "from_id": "...", "to_id": "..."}
{"type": "disconnect_layers", "from_id": "...", "to_id": "..."}
{"type": "reset_architecture"}
```
Server replies to the originating client only with `{"type": "ack", "ok": true}` or `{"type": "error", "message": "..."}`, and — only if the mutation succeeded — broadcasts a fresh `state` message to every connected client, including the originator.

## 13. Frontend spec

- One generic custom node component (`LayerNode.jsx`) parameterized by `data.type`, rendering the type name and a few key params. `input` nodes render with zero input handles; every other node has exactly one input handle and one output handle — including merge nodes, since the diagramming library allows multiple edges into a single target handle natively, so no special multi-handle UI is needed.
- **Auto-layout, not manual dragging-to-position.** Since the shared state has no `x`/`y` (§7), the frontend must recompute layout on every `state` message. Use a simple deterministic layered layout: BFS distance from the `input` node = column index; order within a column = node id order. Write this as a small manual function (`layout.js`) rather than pulling in a layout dependency like `dagre` — good enough for a PoC and one less dependency.
- **Palette** (`LayerPalette.jsx`): a sidebar listing every catalog type. Dragging one onto the canvas sends `add_layer` with that type and its schema defaults; the drop coordinates are discarded (auto-layout takes over on the next state push).
- **Params panel** (`ParamsPanel.jsx`): clicking a node opens a form of its current params (basic text/number inputs per field) with a Save button that sends `update_layer`.
- Dragging an edge between two handles sends `connect_layers`. Selecting a node and pressing delete sends `remove_layer`. A toolbar button sends `reset_architecture`.

## 14. Setup and running

**Backend:**
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py                  # full mode: MCP stdio + websocket server on :8765
MODE=standalone python main.py  # frontend-dev mode: websocket server only
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev   # connects to ws://localhost:8765/ws
```

**Claude Desktop registration** (`claude_desktop_config.snippet.json`), pointing at the venv's interpreter directly since Desktop won't activate your shell environment:
```json
{
  "mcpServers": {
    "nn-architect-mcp": {
      "command": "/absolute/path/to/backend/venv/bin/python",
      "args": ["/absolute/path/to/backend/main.py"]
    }
  }
}
```

## 15. Recommended build order

1. `layers.py` + `codegen.py` + `validator.py`, tested with a hand-written `AppState` dict and plain function calls — no MCP, no websocket, no frontend yet. Confirm a hand-built CNN, a hand-built skip-connection block, and a hand-built transformer block all generate code that actually runs.
2. `store.py` wrapping the above with the 9 operations and their validation/error rules.
3. `mcp_tools.py` + `main.py`'s MCP loop. Test all 9 tools through the MCP Inspector before writing a single line of frontend code.
4. `ws_app.py` + `main.py`'s uvicorn loop, in `standalone` mode, tested with a basic websocket client (or just browser devtools) before the React app exists.
5. Frontend, last — at this point the backend is fully working and the frontend is "just" rendering a state object and sending small messages.

## 16. Acceptance checklist

- [ ] Standalone mode + frontend running with an empty canvas.
- [ ] Claude Desktop connected; ask it to build a small CNN for 28×28 grayscale, 10-class classification — nodes/edges appear live in the browser with no refresh.
- [ ] Ask it to add a residual skip connection (two conv layers in sequence, bypassed by an `add` merge) — confirm codegen and validation both handle it correctly.
- [ ] Ask it to build a small Transformer block (`embedding` → `positional_encoding` → `transformer_encoder_layer` → `flatten`/`linear`) — confirm generated code runs standalone.
- [ ] Ask it to build an LSTM-based sequence model — confirm validation passes.
- [ ] Deliberately mismatch a channel count between two conv layers; call `validate_architecture`; confirm the response names the correct node and layer type; have the agent fix it via `update_layer` and re-validate.
- [ ] In the browser, drag a new layer onto the canvas and connect it manually; confirm the agent's next `get_architecture()` reflects it.
- [ ] Call `generate_code`, save the output to a fresh `.py` file outside the project, run it with nothing but `torch` installed — it runs without modification.
- [ ] Attempt to create a cycle via `connect_layers` — rejected with a clear error.
- [ ] Call `reset_architecture` — canvas clears, next added node is `n1` again.

## 17. Decisions already made for you

Quick-reference list of every non-obvious call, so nothing here needs re-deciding mid-build:

- All tensors are batch-first; sequence tensors `(N, L, E)`, image tensors `(N, C, H, W)`.
- Shape correctness is checked **only** by real execution inside `validate_architecture` — never by static analysis in `add_layer`/`connect_layers`.
- Exactly one `input` node per graph; it can't have incoming edges.
- `concat` input order = edge-insertion order, not node-id order.
- `multihead_attention` is self-attention only (`query=key=value=input`) — no cross-attention.
- RNN/LSTM/GRU: `batch_first=True` always; only the `output` tensor propagates downstream, hidden/cell state is discarded.
- Validation dummy batch size is fixed at N=2.
- Int64 dummy inputs are bounded by the first downstream `embedding`'s `num_embeddings` (default 1000 if none exists).
- Validation errors are structured `LayerExecutionError` objects with `node_id`/`layer_type`/`original` fields — never parsed out of generic error strings.
- Node `(x, y)` position is not part of the shared schema; layout is frontend-only and recomputed every update.
- Backend stdout is reserved exclusively for MCP protocol framing — all logs go to stderr.
- One `ArchitectureStore` is the only mutation path; MCP tools and the websocket handler both call into it, never duplicate logic.
- Orphan nodes (unreachable from `input`) are allowed to exist; they're excluded from codegen/validation with a warning, not a hard error.
- IDs are an incremental counter (`n1`, `n2`, ...), not UUIDs, reset on `reset_architecture`.
- Generated code is fully self-contained (inlines `LayerExecutionError` and `PositionalEncoding`) — no dependency on this project's own backend package.

## 18. Future work (explicitly not built now, noted for later)

- Remote-hosted MCP server (SSE/HTTP transport) with authenticated multi-user sessions, instead of local stdio only.
- Training loop integration — actually fit the designed model to a dataset, with live loss curves.
- Multi-framework export (Keras/TensorFlow/JAX).
- Persistent storage — state currently lives in memory only and is lost on restart.
- Visual design polish / branding beyond default component styling.
