// Deterministic layered layout (spec §13). The shared schema has no x/y, so the
// frontend recomputes positions on every `state` message.
//
// Rules:
//   * column = BFS distance from the input node along directed edges.
//   * a node with no path from input (no input node, or a disconnected island)
//     gets column 0, so it's still visible rather than dropped.
//   * order within a column = node-id order ("n1","n2",... → numeric 1,2,...),
//     which is creation order, so the layout is stable as the graph grows.
//
// Pure: takes {nodes, edges} (backend schema), returns { [nodeId]: {x, y} }.

const COL_GAP = 240;
const ROW_GAP = 110;

// "n12" → 12, for stable numeric ordering. Non-conforming ids sort last by id.
function idOrder(id) {
  const m = /^n(\d+)$/.exec(id);
  return m ? Number(m[1]) : Number.POSITIVE_INFINITY;
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
  // longer path keeps its larger column (max distance), so a layer downstream
  // of a deep branch never sits left of its predecessor.
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

  // Any node not reached above (no input, or disconnected) → column 0.
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
      positions[id] = { x: c * COL_GAP, y: row * ROW_GAP };
    });
  }
  return positions;
}
