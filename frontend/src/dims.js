// Representative "feature width" per layer, used to SIZE its glyph so the shape
// of the network reads at a glance: an autoencoder visibly tapers to its latent,
// a CNN's channels grow, a transformer stays flat. This is purely a rendering
// concern - like x/y positions it is computed on the frontend and never sent to
// the backend (CLAUDE.md invariant 8: geometry/sizes never reach the validator
// or codegen). Pure, so it's testable without mounting React Flow.
//
// "Width" is the channel/feature count a layer emits. Layers that just transform
// their input in place (relu, dropout, norm, pooling, flatten) have no width of
// their own and INHERIT it from upstream, so the tensor's width flows through the
// chain and only the real bottlenecks (linear/conv) change the silhouette.

const prod = (arr) => arr.reduce((a, b) => a * b, 1);

// Width implied by a node's own params, or null if it passes its input straight
// through and should inherit from its predecessor. Adding a new *sized* layer
// type (one that changes the feature count) means adding a case here; pass-through
// types need nothing (they fall to the null default).
export function ownWidth(node) {
  const p = node?.params ?? {};
  switch (node?.type) {
    case "input":
      return Array.isArray(p.shape) ? prod(p.shape) : null;
    case "linear":
      return p.out_features;
    case "conv2d":
    case "conv1d":
    case "conv3d":
    case "conv_transpose2d":
      return p.out_channels;
    case "embedding":
      return p.embedding_dim;
    case "rnn":
    case "lstm":
    case "gru":
      return p.hidden_size != null ? p.hidden_size * (p.bidirectional ? 2 : 1) : null;
    case "multihead_attention":
      return p.embed_dim;
    case "transformer_encoder_layer":
    case "positional_encoding":
      return p.d_model;
    default:
      return null; // activation / norm / pooling / dropout / shape / merge
  }
}

// width for every node, propagating through pass-throughs. concat sums its
// inputs' widths (channels stack); every other multi-input node (add, a plain
// fan-in) takes the max. Cycles and orphan nodes fall back to FALLBACK_WIDTH so
// a width always exists.
export const FALLBACK_WIDTH = 64;

export function computeWidths(arch) {
  const nodes = arch?.nodes ?? [];
  const edges = arch?.edges ?? [];
  if (nodes.length === 0) return {};

  const byId = new Map(nodes.map((n) => [n.id, n]));
  const preds = new Map(nodes.map((n) => [n.id, []]));
  for (const e of edges) {
    if (byId.has(e.from) && byId.has(e.to)) preds.get(e.to).push(e.from);
  }

  const memo = new Map();
  const visiting = new Set();

  const widthOf = (id) => {
    if (memo.has(id)) return memo.get(id);
    if (visiting.has(id)) return FALLBACK_WIDTH; // cycle guard
    visiting.add(id);

    const node = byId.get(id);
    let w = ownWidth(node);
    if (w == null || !Number.isFinite(w) || w <= 0) {
      const ps = preds.get(id) ?? [];
      if (ps.length === 0) {
        w = FALLBACK_WIDTH;
      } else if (node?.type === "concat") {
        w = ps.reduce((sum, p) => sum + widthOf(p), 0);
      } else {
        w = Math.max(...ps.map(widthOf));
      }
    }

    visiting.delete(id);
    memo.set(id, w);
    return w;
  };

  const out = {};
  for (const n of nodes) out[n.id] = widthOf(n.id);
  return out;
}

// Map each node's width to a glyph diameter (px). Normalized PER GRAPH on a sqrt
// scale: the widest layer is maxD, the narrowest minD, everything between by
// sqrt(width) - so the taper is visible regardless of the absolute magnitudes
// (32→784 reads the same as 64→1568). A graph whose layers are all one width
// renders them all at the midpoint rather than implying a bottleneck that isn't
// there.
export const MIN_DIAMETER = 60;
export const MAX_DIAMETER = 128;

export function computeDiameters(arch, { minD = MIN_DIAMETER, maxD = MAX_DIAMETER } = {}) {
  const widths = computeWidths(arch);
  const vals = Object.values(widths);
  if (vals.length === 0) return {};

  const lo = Math.sqrt(Math.min(...vals));
  const hi = Math.sqrt(Math.max(...vals));
  const span = hi - lo;

  const out = {};
  for (const [id, w] of Object.entries(widths)) {
    const t = span === 0 ? 0.5 : (Math.sqrt(w) - lo) / span;
    out[id] = Math.round(minD + t * (maxD - minD));
  }
  return out;
}

// How many representative neuron dots to DRAW for a layer's glyph (used only by the
// linear "neuron-stack" glyph). The count scales with the layer's true feature
// width - per-graph, sqrt-normalized exactly like computeDiameters - so a 768-unit
// layer visibly shows more neurons than a 128 than a 64. It's capped at MAX_NEURONS
// (the glyph then draws top/⋮/bottom to imply the omitted middle) and never exceeds
// the layer's actual neuron count. The *true* count still shows in the dim label,
// so this is a legibility cue, never a claim about the exact number of units.
export const MIN_NEURONS = 3;
export const MAX_NEURONS = 6;

export function computeNeuronCounts(arch, { minN = MIN_NEURONS, maxN = MAX_NEURONS } = {}) {
  const widths = computeWidths(arch);
  const vals = Object.values(widths);
  if (vals.length === 0) return {};

  const lo = Math.sqrt(Math.min(...vals));
  const hi = Math.sqrt(Math.max(...vals));
  const span = hi - lo;

  const out = {};
  for (const [id, w] of Object.entries(widths)) {
    const t = span === 0 ? 0.5 : (Math.sqrt(w) - lo) / span;
    out[id] = Math.min(w, Math.round(minN + t * (maxN - minN)));
  }
  return out;
}

// Geometry of a neuron circle in the EXPANDED "classic neural network" view. Shared
// by the node (LayerNode draws the circles via CSS at this size/gap) and the edge
// (NeuronBundleEdge draws the fully-connected lines between them), so the lines land
// exactly on the circles. A column of `count` circles is centered on the node's
// vertical center, so circle i sits at center + this offset.
//
// When a layer's true width exceeds the circles drawn (`truncated`), the column
// shows a vertical "⋮" for the omitted middle neurons. The ⋮ takes exactly one
// circle-slot (same NEURON_STEP rhythm), so the whole column has `count + 1` slots
// and the circles occupy every slot except the middle one. Both the node's circles
// and the edge's line endpoints are derived from the same slot math below, so the
// gap where the ⋮ sits stays consistent between them.
export const NEURON_CIRCLE = 14; // px, circle diameter
export const NEURON_GAP = 8; // px, vertical gap between circles
export const NEURON_STEP = NEURON_CIRCLE + NEURON_GAP;

// Total vertical slots in the column: `count` circles, plus one for the ⋮ when the
// layer is truncated. The ⋮ occupies the middle slot; index below.
function slotCount(count, truncated) {
  const c = Math.max(1, count | 0);
  return truncated ? c + 1 : c;
}
function ellipsisSlot(count, truncated) {
  return truncated ? Math.floor(slotCount(count, truncated) / 2) : -1;
}

export function neuronYOffsets(count, truncated = false) {
  const slots = slotCount(count, truncated);
  const skip = ellipsisSlot(count, truncated);
  const offsets = [];
  for (let j = 0; j < slots; j++) {
    if (j === skip) continue; // the ⋮ has no circle → no edge endpoint here
    offsets.push((j - (slots - 1) / 2) * NEURON_STEP);
  }
  return offsets;
}

// How many circles render above the ⋮ (the rest go below). Used by LayerNode so its
// DOM order matches the slot layout the offsets assume.
export function neuronSplit(count, truncated) {
  const c = Math.max(1, count | 0);
  if (!truncated) return { top: c, bottom: 0 };
  const top = ellipsisSlot(count, truncated); // circles fill slots [0..mid-1]
  return { top, bottom: c - top };
}

// Half the pixel height of a `count`-circle column (including the ⋮ slot when
// truncated) - LayerNode uses it to place the type/dim labels just below the column
// (which can be taller than the node cell).
export function neuronColumnHalfHeight(count, truncated = false) {
  const slots = slotCount(count, truncated);
  return (slots * NEURON_CIRCLE + (slots - 1) * NEURON_GAP) / 2;
}
