// Map the backend architecture ({nodes:[{id,type,params}], edges:[{from,to}]})
// into React Flow elements. Node rendering is delegated to the single "layer"
// node type (LayerNode). Positions are a placeholder here - App assigns real
// ones from the column placement in layout.js (stable across broadcasts). Pure,
// so it's testable without mounting React Flow.

import { categoryOf } from "./catalog.js";

export function toFlowElements(arch) {
  const nodes = arch?.nodes ?? [];
  const edges = arch?.edges ?? [];

  const flowNodes = nodes.map((n) => ({
    id: n.id,
    type: "layer",
    position: { x: 0, y: 0 }, // overwritten by App from layout.js placement
    // `category`/`label`/`synthetic` are honored when present (expansion sub-nodes
    // carry them); a normal broadcast node has none, so category falls back to the
    // catalog and label/synthetic stay undefined - identical to the pre-expansion
    // behavior.
    data: {
      type: n.type,
      params: n.params,
      category: n.category ?? categoryOf(n.type),
      label: n.label,
      synthetic: n.synthetic,
    },
  }));

  const flowEdges = edges.map((e) => ({
    id: `${e.from}->${e.to}`,
    source: e.from,
    target: e.to,
  }));

  return { nodes: flowNodes, edges: flowEdges };
}

// Guard for React Flow's `onNodesChange`/`onEdgesChange` streams. The backend is
// the single source of truth for *which* nodes/edges exist (CLAUDE.md invariant
// 2): the only ways something is removed are the user's explicit delete (which
// goes to the backend and comes back as a broadcast) and the broadcast itself.
// React Flow, however, also emits `remove` changes from its own internal
// bookkeeping - e.g. when a freshly-replaced node hasn't been re-measured yet and
// an edge's handle can't be resolved for a frame. Applying those to our
// controlled state silently deletes elements the backend still has, so the canvas
// desyncs (edges vanish while `generate_code` still works). We drop `remove`
// changes and let geometry/selection (`position`/`dimensions`/`select`) through;
// real deletions still flow via `onNodesDelete`/`onEdgesDelete` -> backend ->
// broadcast, so nothing user-initiated is lost - it just round-trips the store.
export function dropRemoveChanges(changes) {
  return changes.filter((c) => c.type !== "remove");
}
