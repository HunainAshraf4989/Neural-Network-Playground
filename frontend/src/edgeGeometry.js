// Pure geometry for the canvas edges, split out of DeletableEdge.jsx so it's
// unit-testable without importing React Flow.

import { COL_GAP } from "./layout.js";

// An edge spanning more than ~one column horizontally is a skip/residual: its
// endpoints are not neighbors, so it should arc over the layers between them
// rather than cut straight through. (Adjacent handles are barely a fraction of a
// column apart; a one-node jump is already well over COL_GAP.)
export function isSkipSpan(sourceX, targetX, colGap = COL_GAP) {
  return targetX - sourceX > colGap;
}

// A cubic that bows up and over the layers between source and target — the
// conventional residual/skip arc. Lift grows with the span so longer skips clear
// more nodes.
export function skipArc(sourceX, sourceY, targetX, targetY) {
  const dx = targetX - sourceX;
  const lift = Math.min(180, 60 + Math.abs(dx) * 0.16);
  const apexY = Math.min(sourceY, targetY) - lift;
  const path = `M${sourceX},${sourceY} C${sourceX + dx * 0.2},${apexY} ${targetX - dx * 0.2},${apexY} ${targetX},${targetY}`;
  return { path, labelX: (sourceX + targetX) / 2, labelY: apexY + 9 };
}
