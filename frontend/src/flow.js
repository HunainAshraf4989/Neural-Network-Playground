// Map the backend architecture ({nodes:[{id,type,params}], edges:[{from,to}]})
// into React Flow elements. Node rendering is delegated to the single "layer"
// node type (LayerNode). Positions are a placeholder here — App assigns real
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
    data: { type: n.type, params: n.params, category: categoryOf(n.type) },
  }));

  const flowEdges = edges.map((e) => ({
    id: `${e.from}->${e.to}`,
    source: e.from,
    target: e.to,
  }));

  return { nodes: flowNodes, edges: flowEdges };
}
