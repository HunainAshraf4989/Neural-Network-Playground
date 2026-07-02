// "Expand architecture" — a FRONTEND-ONLY view transform that substitutes a
// composite macro-layer with a read-only subgraph showing its real PyTorch
// internals. It mirrors what codegen.py emits for each type, so the expanded
// picture is truthful (a transformer_encoder_layer becomes self-attn + residual
// + norm + FFN + residual + norm, exactly as nn.TransformerEncoderLayer runs).
//
// Like positions and glyph sizes (dims.js), this never touches the shared schema:
// it runs purely on the frontend AFTER a broadcast and is never sent back, so the
// store / validator / codegen never see the sub-nodes (CLAUDE.md invariant 8). The
// result has the same {nodes, edges, layout} shape as a broadcast, so the existing
// layout / sizing / glyph pipeline renders it unchanged.
//
// Sub-nodes reuse REAL catalog types wherever one exists (linear, layernorm, relu,
// dropout, add, concat, multihead_attention, rnn/lstm/gru) so they size and draw
// for free; the one construct with no catalog equivalent — the attention core —
// carries an explicit `category` instead. Every sub-node is flagged `synthetic`
// so the canvas renders it dimmed and inert (no drag / connect / delete / edit).
//
// Pure, so it's unit-testable without mounting React Flow (see expansions.test.js).

// Top-level types that have an internal structure worth showing.
export const EXPANDABLE_TYPES = new Set([
  "transformer_encoder_layer",
  "multihead_attention",
  "rnn",
  "lstm",
  "gru",
]);

// Build one synthetic sub-node. `category` is only needed for the (single) type
// with no catalog entry; for real catalog types it's left undefined so the flow
// adapter derives it from the type like any normal node.
function sub(parentId, name, type, params, { label, category } = {}) {
  return {
    id: `${parentId}/${name}`,
    type,
    params,
    synthetic: true,
    label: label ?? name,
    ...(category ? { category } : {}),
  };
}

// nn.TransformerEncoderLayer (post-norm default, activation=relu) — the showcase.
// Forward (from codegen.py:125 -> torch):
//   x = norm1(x + dropout(self_attn(x, x, x)))          # self-attention block
//   x = norm2(x + dropout(linear2(dropout(relu(linear1(x))))))   # feed-forward block
// The two residual skips fan the block input into an `add`, which the canvas draws
// as a dashed arc over the block.
function expandTransformer(node) {
  const id = node.id;
  const p = node.params;
  const d = p.d_model;
  const nodes = [
    sub(id, "attn", "multihead_attention", { embed_dim: d, num_heads: p.nhead, dropout: p.dropout }, { label: "self-attn" }),
    sub(id, "do1", "dropout", { p: p.dropout }, { label: "dropout" }),
    sub(id, "res1", "add", {}, { label: "+ residual" }),
    sub(id, "norm1", "layernorm", { normalized_shape: d }, { label: "norm1" }),
    sub(id, "fc1", "linear", { in_features: d, out_features: p.dim_feedforward }, { label: "linear1" }),
    sub(id, "act", "relu", {}, { label: "relu" }),
    sub(id, "do2", "dropout", { p: p.dropout }, { label: "dropout" }),
    sub(id, "fc2", "linear", { in_features: p.dim_feedforward, out_features: d }, { label: "linear2" }),
    sub(id, "do3", "dropout", { p: p.dropout }, { label: "dropout" }),
    sub(id, "res2", "add", {}, { label: "+ residual" }),
    sub(id, "norm2", "layernorm", { normalized_shape: d }, { label: "norm2" }),
  ];
  const E = (n) => `${id}/${n}`;
  const edges = [
    { from: E("attn"), to: E("do1") },
    { from: E("do1"), to: E("res1") },
    { from: E("res1"), to: E("norm1") },
    { from: E("norm1"), to: E("fc1") },
    { from: E("fc1"), to: E("act") },
    { from: E("act"), to: E("do2") },
    { from: E("do2"), to: E("fc2") },
    { from: E("fc2"), to: E("do3") },
    { from: E("do3"), to: E("res2") },
    { from: E("res2"), to: E("norm2") },
    { from: E("norm1"), to: E("res2") }, // feed-forward residual skip
  ];
  // The block input feeds self-attn AND the first residual add (attention skip).
  return { nodes, edges, entries: [E("attn"), E("res1")], exit: E("norm2") };
}

// nn.MultiheadAttention (self-attention) — Q/K/V projections into the scaled
// dot-product attention core (over `num_heads` heads), then the output projection.
function expandMultiheadAttention(node) {
  const id = node.id;
  const p = node.params;
  const d = p.embed_dim;
  const nodes = [
    sub(id, "q", "linear", { in_features: d, out_features: d }, { label: "Q proj" }),
    sub(id, "k", "linear", { in_features: d, out_features: d }, { label: "K proj" }),
    sub(id, "v", "linear", { in_features: d, out_features: d }, { label: "V proj" }),
    sub(id, "sdpa", "scaled_dot_product_attention", { num_heads: p.num_heads, dropout: p.dropout },
      { label: "SDPA", category: "attention" }),
    sub(id, "out", "linear", { in_features: d, out_features: d }, { label: "out proj" }),
  ];
  const E = (n) => `${id}/${n}`;
  const edges = [
    { from: E("q"), to: E("sdpa") },
    { from: E("k"), to: E("sdpa") },
    { from: E("v"), to: E("sdpa") },
    { from: E("sdpa"), to: E("out") },
  ];
  return { nodes, edges, entries: [E("q"), E("k"), E("v")], exit: E("out") };
}

// A stacked / bidirectional RNN (rnn|lstm|gru) unrolls into its per-layer cells.
// Only meaningful when there's actually more than one cell — a plain 1-layer
// unidirectional RNN has nothing graph-level to show (gate internals are below
// this canvas's altitude), so it returns null and is left as a single node.
function expandRecurrent(node) {
  const p = node.params;
  const layers = p.num_layers ?? 1;
  const bidi = !!p.bidirectional;
  if (layers <= 1 && !bidi) return null;

  const id = node.id;
  const type = node.type;
  const h = p.hidden_size;
  const E = (n) => `${id}/${n}`;
  const cell = (name, inSize, label) =>
    sub(id, name, type, { input_size: inSize, hidden_size: h, num_layers: 1, bidirectional: false }, { label });

  const nodes = [];
  const edges = [];
  let prevExit = null; // exit id of the previous layer
  let entries = null; // entries of the first layer
  for (let i = 0; i < layers; i++) {
    const inSize = i === 0 ? p.input_size : h * (bidi ? 2 : 1);
    let layerEntries;
    let layerExit;
    if (bidi) {
      const fwd = cell(`l${i}f`, inSize, `${type} L${i + 1} →`);
      const bwd = cell(`l${i}b`, inSize, `${type} L${i + 1} ←`);
      const cat = sub(id, `l${i}c`, "concat", { dim: -1 }, { label: "concat" });
      nodes.push(fwd, bwd, cat);
      edges.push({ from: fwd.id, to: cat.id }, { from: bwd.id, to: cat.id });
      layerEntries = [fwd.id, bwd.id];
      layerExit = cat.id;
    } else {
      const c = cell(`l${i}`, inSize, `${type} L${i + 1}`);
      nodes.push(c);
      layerEntries = [c.id];
      layerExit = c.id;
    }
    if (prevExit) for (const t of layerEntries) edges.push({ from: prevExit, to: t });
    if (entries == null) entries = layerEntries;
    prevExit = layerExit;
  }
  return { nodes, edges, entries, exit: prevExit };
}

const EXPANDERS = {
  transformer_encoder_layer: expandTransformer,
  multihead_attention: expandMultiheadAttention,
  rnn: expandRecurrent,
  lstm: expandRecurrent,
  gru: expandRecurrent,
};

// The subgraph template for a node, or null if this node has nothing to expand.
export function expandNode(node) {
  const fn = EXPANDERS[node?.type];
  return fn ? fn(node) : null;
}

// Whether the "Expand" button would actually change this node.
export function isExpandable(node) {
  return expandNode(node) != null;
}

// Whether the node can be UNGROUPED — persistently decomposed into real store
// nodes by the backend (unlike Expand, a read-only view). Same families as
// isExpandable EXCEPT plain multihead_attention: its expansion needs the
// synthetic scaled_dot_product_attention type, which isn't in the catalog yet.
export function isUngroupable(node) {
  return node?.type !== "multihead_attention" && isExpandable(node);
}

// Substitute every expandable node in `arch` with its internal subgraph, rewiring
// external edges to the subgraph's entry/exit, and return a graph of the same
// {nodes, edges, layout} shape. Non-expandable nodes pass through untouched.
export function expandGraph(arch) {
  const nodes = arch?.nodes ?? [];
  const edges = arch?.edges ?? [];

  const expanded = new Map(); // parentId -> {entries:[id], exit:id}
  const outNodes = [];
  const outEdges = [];

  for (const n of nodes) {
    const tpl = expandNode(n);
    if (tpl) {
      expanded.set(n.id, { entries: tpl.entries, exit: tpl.exit });
      outNodes.push(...tpl.nodes);
      outEdges.push(...tpl.edges);
    } else {
      outNodes.push(n);
    }
  }

  for (const e of edges) {
    const fromIds = expanded.has(e.from) ? [expanded.get(e.from).exit] : [e.from];
    const toIds = expanded.has(e.to) ? expanded.get(e.to).entries : [e.to];
    for (const from of fromIds) for (const to of toIds) outEdges.push({ from, to });
  }

  return { nodes: outNodes, edges: outEdges, layout: arch?.layout };
}
