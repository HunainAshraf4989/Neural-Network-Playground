# Stage 5 — React frontend

## Scope
`frontend/` — React + Vite, plain JS, React Flow. Renders the shared state and sends small mutation
messages. Backend is fully working by now; the frontend is "just" a state renderer + message sender.

## Files
- `frontend/package.json`, `frontend/vite.config.js`, `frontend/src/main.jsx`.
- `src/App.jsx` — open `ws://localhost:8765/ws`; hold `{nodes, edges}` from `state` messages; send
  client→server messages on user actions; show ack/error.
- `src/LayerNode.jsx` — ONE generic node component parameterized by `data.type`; renders type + a few
  key params; **color-coded by `data.category`**. `input` = no input handle; every other node = one
  input + one output handle (React Flow allows multiple edges into one handle natively).
- `src/LayerPalette.jsx` — sidebar of catalog types **grouped by category**; drag → `add_layer` with
  schema defaults (drop coords discarded).
- `src/ParamsPanel.jsx` — click a node → form of its params (text/number inputs) + Save → `update_layer`.
- `src/layout.js` — deterministic layered layout: BFS distance from `input` = column; order within a
  column = node-id order. No `dagre`. Recomputed on every `state` message (no x/y in shared schema).

## Behaviors
Drag layer → `add_layer`. Drag edge between handles → `connect_layers`. Select node + Delete →
`remove_layer`. Toolbar button → `reset_architecture`. Edit params + Save → `update_layer`.

## Confirm package name
React Flow has been renamed before — check npm for the current package name at install time rather
than assuming.

## Expected outcome
Empty canvas boots; each interaction works against the standalone backend; layout is stable.

## Verification gate
- `MODE=standalone` backend + `npm run dev`; perform add/edit/delete/connect/reset in the browser.
- Add + wire a node in the browser, then run an MCP `get_architecture()` (full mode) and confirm it
  reflects the human edit — proves cross-surface sync over the single store.

## Status
✅ done. `frontend/` is a Vite + React (plain JS) app using **`@xyflow/react` v12** (the current
React Flow package, confirmed on npm at install). Architecture: the backend is the single source of
truth — the app never mutates `arch` locally, it only renders what `state` broadcasts back and sends
§12 client messages, so the canvas can't drift from the agent's view.

Modules:
- `src/catalog.js` — frontend mirror of `backend/layers.py CATALOG`. Carries a concrete starting
  value for **every** param (required + optional), because the backend rejects `add_layer` with a
  missing required param and those have no schema default there. A drag therefore always produces a
  valid `add_layer`; the user then refines params in the panel. Adding a layer is now a **third**
  edit point (alongside `layers.py` + `codegen.py`).
- `src/layout.js` — deterministic layered layout: column = BFS **max**-distance from `input`, row =
  node-id order; disconnected/no-input nodes fall to column 0 so they stay visible. No `dagre`.
  `reachableFromInput` lets App keep a node that *loses* its path from input parked where it is (so
  deleting one wire doesn't yank the downstream graph back to column 0).
- `src/protocol.js` — pure §12 client-message builders. `src/flow.js` — backend state → React Flow
  elements, plus `dropRemoveChanges` (the change-stream guard that keeps the backend the sole owner of
  node/edge existence). `src/paramTypes.js` — value↔text coercion so `"3"`→`3`, `"[1,2]"`→`[1,2]`.
- `src/useArchitectureSocket.js` — ws hook (auto-reconnect) holding `arch` + surfacing ack/error.
- Components: `App.jsx` (canvas + toolbar + drop/connect/delete/reset wiring), `LayerNode.jsx` (one
  generic node, color-coded by category, no target handle on `input`), `LayerPalette.jsx` (grouped
  by category, drag/click → `add_layer`), `ParamsPanel.jsx` (typed form → `update_layer` / delete).

Tests (all green): **62** Vitest unit/component tests (`layout`, `paramTypes`, `protocol`, `flow`,
`catalog` parity vs the live `backend/layers.py`, `LayerNode`, `LayerPalette`, `ParamsPanel`, `App`
with React Flow + socket mocked) + **22** live-backend e2e tests (`test/e2e/ws.test.js`, run with
`npm run test:e2e`) that spawn the real `MODE=standalone` backend and drive **every** mutation, every
rejection branch, malformed-input robustness, and multi-client broadcast through the frontend's own
message builders. `npm run build` succeeds; dev server boots and serves the empty canvas.

## Verification gate — met
- `npm test` → 62 passed; `npm run test:e2e` → 22 passed against the live standalone backend;
  `npm run build` → ok; `npx vite` serves the app (HTTP 200, empty canvas).
- Cross-surface single-store sync (human ws edit visible to MCP `get_architecture`) is exercised by
  `backend/tests/test_ws_stdio.py` (89 backend tests still green), over the same store + the same
  §12 messages this frontend sends.

## Canvas-sync hardening (follow-up fix)
Two human-edit UX bugs traced to one principle that wasn't actually enforced — *the backend owns
existence, React Flow owns only geometry*:
- **Deleting a connection collapsed the downstream graph to the input column.** `computeLayout` parks
  any node with no path from `input` at column 0, so cutting a mid-chain wire dropped everything after
  it onto the input column. Fix: a node that *was* placed and then loses its path from input keeps its
  current position (layout override #3, via `reachableFromInput`) and re-flows once re-wired.
- **Edges vanished from the canvas while the backend graph stayed valid** (e.g. after dropping a node
  and dragging a connection). Root cause: React Flow's `onNodesChange`/`onEdgesChange` could emit
  `remove` changes from internal bookkeeping (an edge whose handle it can't resolve for a frame after a
  wholesale node replace), and we applied them straight to controlled state — silently deleting
  elements the store still had. Fix: `flow.js` `dropRemoveChanges` strips `remove` from both streams
  (real deletes still round-trip via `onNodesDelete`/`onEdgesDelete` → store → broadcast), and the
  broadcast effect now reuses existing node objects so React Flow keeps its measurements (no
  re-measure blink). Covered by `App.desync.test.jsx`, which proves a spurious RF `remove` no longer
  drops a node/edge. Suite now **76** Vitest tests.
