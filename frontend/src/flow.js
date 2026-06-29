// Map the backend architecture ({nodes:[{id,type,params}], edges:[{from,to}]})
// into React Flow elements. Positions come from the deterministic layout; node
// rendering is delegated to the single "layer" node type (LayerNode). Pure, so
// it's testable without mounting React Flow.

import { computeLayout } from "./layout.js";
import { categoryOf } from "./catalog.js";

export function toFlowElements(arch) {
  const nodes = arch?.nodes ?? [];
  const edges = arch?.edges ?? [];
  const positions = computeLayout(arch);

  const flowNodes = nodes.map((n) => ({
    id: n.id,
    type: "layer",
    position: positions[n.id] ?? { x: 0, y: 0 },
    data: { type: n.type, params: n.params, category: categoryOf(n.type) },
  }));

  const flowEdges = edges.map((e) => ({
    id: `${e.from}->${e.to}`,
    source: e.from,
    target: e.to,
  }));

  return { nodes: flowNodes, edges: flowEdges };
}
