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
⬜ not started.
