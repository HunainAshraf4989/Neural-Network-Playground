// One generic node for every layer type (spec §13), parameterized by `data.type`
// and color-coded by `data.category`. Rendered as a circle so the graph reads
// like a neural-network diagram. The `input` type has no target handle; every
// other type has exactly one target + one source handle. React Flow allows
// multiple edges into a single target handle, so merge nodes (add/concat) need no
// special UI. Params are hidden on the circle and revealed in a hover tooltip
// (the full, editable set lives in the params panel).

import { Handle, Position } from "@xyflow/react";
import { colorOf } from "./catalog.js";

// Show a few key params in the hover tooltip; the full set lives in the panel.
// `null` (e.g. pooling stride) renders as "auto".
const MAX_INLINE_PARAMS = 4;

function formatValue(v) {
  if (v === null) return "auto";
  if (Array.isArray(v)) return `[${v.join(",")}]`;
  return String(v);
}

export default function LayerNode({ data, selected }) {
  const color = colorOf(data.type);
  const params = Object.entries(data.params ?? {}).slice(0, MAX_INLINE_PARAMS);

  return (
    <div className="nn-node" title={data.type}>
      {data.type !== "input" && (
        <Handle type="target" position={Position.Left} className="nn-handle" />
      )}

      <div
        className={`nn-node__circle${selected ? " is-selected" : ""}`}
        style={{ "--node-color": color }}
      >
        <span className="nn-node__label">{data.type}</span>
      </div>

      {params.length > 0 && (
        <div className="nn-node__tooltip">
          {params.map(([k, v]) => (
            <div key={k} className="nn-node__tooltip-row">
              <span className="nn-node__tooltip-key">{k}</span>
              <span className="nn-node__tooltip-val">{formatValue(v)}</span>
            </div>
          ))}
        </div>
      )}

      <Handle type="source" position={Position.Right} className="nn-handle" />
    </div>
  );
}
