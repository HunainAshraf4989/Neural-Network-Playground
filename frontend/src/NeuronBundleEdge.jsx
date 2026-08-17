// The EXPANDED-view edge between two neuron columns: the fully-connected lines of a
// classic deep-net diagram. Where DeletableEdge draws one line node-to-node, this
// draws sourceCount × targetCount straight lines, from each source neuron circle to
// each target neuron circle.
//
// Alignment: circles are centered on the node's vertical center, and React Flow gives
// us the handle position (sourceX/Y, targetX/Y) at that same center. So circle i sits
// at sourceY + neuronYOffsets(count)[i] - the exact offsets the node renders with (both
// import them from dims.js), which is what makes the lines land on the circles.
//
// Read-only by design (the expanded view doesn't edit structure), so there's no hover
// ✕ - the real connection is edited back in the collapsed view.

import { neuronYOffsets } from "./dims.js";

export default function NeuronBundleEdge({ sourceX, sourceY, targetX, targetY, data }) {
  const sc = data?.sourceCount ?? 1;
  const tc = data?.targetCount ?? 1;
  // Truncated columns skip the middle slot (the ⋮), so the endpoints must use the
  // same slot math the node's circles do - otherwise lines miss the circles.
  const sourceYs = neuronYOffsets(sc, data?.sourceTruncated).map((o) => sourceY + o);
  const targetYs = neuronYOffsets(tc, data?.targetTruncated).map((o) => targetY + o);

  // One <path> holding every M…L segment - one DOM node for the whole bundle.
  const segments = [];
  for (const sy of sourceYs) {
    for (const ty of targetYs) {
      segments.push(`M${sourceX},${sy} L${targetX},${ty}`);
    }
  }

  // Concrete hex (not a CSS var) so the lines survive html-to-image's PNG export,
  // which clones a detached viewport and loses var-based colors (see DeletableEdge).
  return <path d={segments.join(" ")} className="nn-bundle" fill="none" stroke="#5b8def" />;
}
