// Edge with an always-visible "×" button at its midpoint, so a connection can be
// removed with a single click (no need to discover the select-then-Backspace
// gesture). The button calls `data.onDelete`, which App wires to a
// `disconnect_layers` message — the backend stays the single source of truth.

import { BaseEdge, EdgeLabelRenderer, getBezierPath } from "@xyflow/react";

export default function DeletableEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  style,
  data,
}) {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <>
      <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} style={style} />
      <EdgeLabelRenderer>
        <button
          type="button"
          className="nn-edge-delete nodrag nopan"
          style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
          onClick={(e) => {
            e.stopPropagation();
            data?.onDelete?.();
          }}
          title="Delete connection"
          aria-label="Delete connection"
        >
          ×
        </button>
      </EdgeLabelRenderer>
    </>
  );
}
