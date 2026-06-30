// Deterministic edge-driven layout (the shared schema has no x/y, so positions
// are recomputed from the graph on every broadcast).
//
// Rules:
//   * column = longest directed distance from the input node. So adding an edge
//     n1 -> n2 puts n2 one column to the RIGHT of n1, and a merge sits right of
//     its deepest input. This is what makes wiring a network lay itself out.
//   * the input node is column 0 (leftmost); a node with no path from input
//     (no input yet, or a not-yet-wired node) gets column 0 too, so it stays
//     visible rather than vanishing.
//   * row = order within a column by node id (creation order), so the layout is
//     stable as the graph grows.
//
// App layers two overrides on top of this base (see App.jsx): agent placement
// hints (the backend `layout` channel) and the user's own drag positions.
// Pure: takes {nodes, edges}, returns { [nodeId]: {x, y} }.

export const COL_GAP = 120; // column step (px) — a touch wider than a node circle (80)
export const ROW_GAP = 110; // row step (px)

// "n12" → 12, for stable numeric ordering. Non-conforming ids sort last by id.
function idOrder(id) {
  const m = /^n(\d+)$/.exec(id);
  return m ? Number(m[1]) : Number.POSITIVE_INFINITY;
}

export const colOf = (p) => Math.round(p.x / COL_GAP);
export const rowOf = (p) => Math.round(p.y / ROW_GAP);
export const cellPos = (col, row) => ({ x: col * COL_GAP, y: row * ROW_GAP });

// Snap an arbitrary point onto the nearest grid cell (clamped to col/row ≥ 0 so
// nothing lands left of the input column or above the top row).
export const snapToCell = (p) =>
  cellPos(Math.max(0, Math.round(p.x / COL_GAP)), Math.max(0, Math.round(p.y / ROW_GAP)));

// Set of node ids with a directed path from the input node (the input itself
// included). Empty when there is no input node. App uses this to keep a node
// that *loses* its path from input (e.g. the user deleted an upstream edge)
// parked at its current position instead of collapsing it back to the input
// column — see the "sticky" rule in App.jsx. Pure; ignores edges to/from
// unknown nodes (mirrors computeLayout).
export function reachableFromInput(arch) {
  const nodes = arch?.nodes ?? [];
  const edges = arch?.edges ?? [];
  const input = nodes.find((n) => n.type === "input");
  if (!input) return new Set();

  const ids = new Set(nodes.map((n) => n.id));
  const adj = new Map();
  for (const e of edges) {
    if (ids.has(e.from) && ids.has(e.to)) {
      if (!adj.has(e.from)) adj.set(e.from, []);
      adj.get(e.from).push(e.to);
    }
  }

  const seen = new Set([input.id]);
  const stack = [input.id];
  while (stack.length) {
    const id = stack.pop();
    for (const next of adj.get(id) ?? []) {
      if (!seen.has(next)) {
        seen.add(next);
        stack.push(next);
      }
    }
  }
  return seen;
}

export function computeLayout(arch) {
  const nodes = arch?.nodes ?? [];
  const edges = arch?.edges ?? [];
  if (nodes.length === 0) return {};

  const ids = new Set(nodes.map((n) => n.id));
  const adj = new Map(); // from -> [to...]
  for (const e of edges) {
    if (ids.has(e.from) && ids.has(e.to)) {
      if (!adj.has(e.from)) adj.set(e.from, []);
      adj.get(e.from).push(e.to);
    }
  }

  // BFS column assignment from the input node (if any). A node reached by a
  // longer path keeps its larger column (max distance), so a node downstream of
  // a deep branch never sits left of its predecessor.
  const col = new Map();
  const input = nodes.find((n) => n.type === "input");
  if (input) {
    const queue = [[input.id, 0]];
    col.set(input.id, 0);
    while (queue.length) {
      const [id, d] = queue.shift();
      for (const next of adj.get(id) ?? []) {
        const nd = d + 1;
        if (!col.has(next) || nd > col.get(next)) {
          col.set(next, nd);
          queue.push([next, nd]);
        }
      }
    }
  }

  // Any node not reached above (no input, or not yet wired) → column 0.
  for (const n of nodes) {
    if (!col.has(n.id)) col.set(n.id, 0);
  }

  // Group node ids by column, order each column by node id.
  const byCol = new Map();
  for (const n of nodes) {
    const c = col.get(n.id);
    if (!byCol.has(c)) byCol.set(c, []);
    byCol.get(c).push(n.id);
  }

  const positions = {};
  for (const [c, idsInCol] of byCol) {
    idsInCol.sort((a, b) => idOrder(a) - idOrder(b) || (a < b ? -1 : 1));
    idsInCol.forEach((id, row) => {
      positions[id] = cellPos(c, row);
    });
  }
  return positions;
}
