// Representative "feature width" per layer, used to SIZE its glyph so the shape
// of the network reads at a glance: an autoencoder visibly tapers to its latent,
// a CNN's channels grow, a transformer stays flat. This is purely a rendering
// concern — like x/y positions it is computed on the frontend and never sent to
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
// sqrt(width) — so the taper is visible regardless of the absolute magnitudes
// (32→784 reads the same as 64→1568). A graph whose layers are all one width
// renders them all at the midpoint rather than implying a bottleneck that isn't
// there.
export const MIN_DIAMETER = 44;
export const MAX_DIAMETER = 100;

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
