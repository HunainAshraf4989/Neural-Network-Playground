// Edge that reveals a delete control only on hover (or when selected), instead of
// a permanently-visible button cluttering every connection. Hovering highlights
// the path and fades in a small trash button at its midpoint; clicking it calls
// `data.onDelete`, which App wires to a `disconnect_layers` message — the backend
// stays the single source of truth. A wide transparent overlay path makes the
// thin edge easy to hover/click.

import { useState } from "react";
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
  selected,
  data,
}) {
  const [hovered, setHovered] = useState(false);
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const active = hovered || selected;

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{ ...style, stroke: active ? "var(--accent)" : undefined, strokeWidth: active ? 2.5 : undefined }}
      />
      {/* Invisible wide hit area so the thin edge is easy to hover/click. */}
      <path
        d={edgePath}
        fill="none"
        stroke="transparent"
        strokeWidth={20}
        className="nn-edge-hit"
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      />
      <EdgeLabelRenderer>
        <button
          type="button"
          className={`nn-edge-delete nodrag nopan${active ? " is-visible" : ""}`}
          style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
          onMouseEnter={() => setHovered(true)}
          onMouseLeave={() => setHovered(false)}
          onClick={(e) => {
            e.stopPropagation();
            data?.onDelete?.();
          }}
          title="Delete connection"
          aria-label="Delete connection"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6M10 11v6M14 11v6" />
          </svg>
        </button>
      </EdgeLabelRenderer>
    </>
  );
}
