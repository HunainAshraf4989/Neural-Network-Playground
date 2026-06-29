// One generic node for every layer type (spec §13), parameterized by `data.type`
// and color-coded by `data.category`. The `input` type has no target handle;
// every other type has exactly one target + one source handle. React Flow allows
// multiple edges into a single target handle, so merge nodes (add/concat) need no
// special UI.

import { Handle, Position } from "@xyflow/react";
import { colorOf } from "./catalog.js";

// Show a few key params inline so the node is readable at a glance; the full set
// lives in the params panel. `null` (e.g. pooling stride) renders as "auto".
const MAX_INLINE_PARAMS = 3;

function formatValue(v) {
  if (v === null) return "auto";
  if (Array.isArray(v)) return `[${v.join(",")}]`;
  return String(v);
}

export default function LayerNode({ data, selected }) {
  const color = colorOf(data.type);
  const params = Object.entries(data.params ?? {}).slice(0, MAX_INLINE_PARAMS);

  return (
    <div
      className="layer-node"
      style={{ borderColor: color, boxShadow: selected ? `0 0 0 2px ${color}` : "none" }}
    >
      {data.type !== "input" && <Handle type="target" position={Position.Left} />}
      <div className="layer-node__title" style={{ color }}>
        {data.type}
      </div>
      {params.length > 0 && (
        <div className="layer-node__params">
          {params.map(([k, v]) => (
            <div key={k} className="layer-node__param">
              <span className="layer-node__param-key">{k}</span>
              <span className="layer-node__param-val">{formatValue(v)}</span>
            </div>
          ))}
        </div>
      )}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
