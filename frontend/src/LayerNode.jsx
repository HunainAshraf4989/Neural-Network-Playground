// One node per layer (spec §13), but its GLYPH is chosen by category so the graph
// reads like the conventional diagram for whatever was built — a CNN's conv slabs,
// an MLP's neuron stacks, a transformer's blocks — and its SIZE scales with the
// layer's feature width (see dims.js) so the network's silhouette is legible: an
// autoencoder visibly pinches at its latent, a CNN's channels swell. The idiom is
// derived from the graph, never toggled.
//
// Invariants kept across all glyphs: the `input` type has no target handle, every
// other type has one target + one source handle (React Flow allows many edges into
// one target handle, so merge nodes need no special wiring); the type name shows on
// the glyph and the key params live in a hover tooltip (full set in the panel).

import { Handle, Position } from "@xyflow/react";
import { colorOfCategory } from "./catalog.js";

const MAX_INLINE_PARAMS = 4;
const DEFAULT_DIAMETER = 76; // used when a node is rendered without a computed size

// In-place ops (activation/norm/dropout/flatten/pooling) operate on whatever
// tensor flows through them — they aren't representations, so they render at a
// fixed small size and stay "quiet", letting the structural layers (linear/conv/…)
// carry the silhouette instead of a giant relu dwarfing its own linear.
const QUIET_CATEGORIES = new Set(["activation", "norm", "regularization", "shape", "pooling"]);
const QUIET_DIAMETER = 46;

// The node sits in a fixed-size cell (matches .nn-node in styles.css) with the
// glyph centered inside, so connection handles must be INSET to the glyph's real
// edge — otherwise they float in the empty cell, away from the visible shape, and
// the graph becomes fiddly to wire by hand. Each family's width relative to `--d`:
const NODE_BOX = 116;
const WIDTH_FACTOR = { stack: 0.72, block: 1.18, loop: 1.18, merge: 0.82, tag: 1.12 };

function formatValue(v) {
  if (v === null) return "auto";
  if (Array.isArray(v)) return `[${v.join(",")}]`;
  return String(v);
}

// Glyph family per category — the visual vocabulary that makes each NN family
// recognizable. Everything not called out (activation/norm/pooling/dropout/
// shape/embedding) is a plain circle, the quiet "in-place op" node.
function glyphFamily(category) {
  switch (category) {
    case "conv": return "slab"; // stacked feature-map planes
    case "linear": return "stack"; // a column of neurons (the FC layer)
    case "recurrent": return "loop"; // a unit with recurrence
    case "attention": return "block"; // MHA/FFN composite block
    case "merge": return "merge"; // ⊕ / ∥ junction
    case "input": return "tag"; // the data tensor
    default: return "circle";
  }
}

function mergeSign(type) {
  return type === "concat" ? "∥" : "+";
}

function blockSub(type) {
  if (type === "multihead_attention") return "MHA";
  if (type === "transformer_encoder_layer") return "MHA·FFN";
  if (type === "positional_encoding") return "POS";
  return null;
}

export default function LayerNode({ data, selected }) {
  // Color from category (not type) so expansion sub-nodes with a synthetic type —
  // e.g. the attention core — still color correctly; for a normal node the
  // category IS the type's category, so this is unchanged.
  const color = colorOfCategory(data.category);
  const label = data.label ?? data.type; // sub-nodes carry a friendlier label
  const params = Object.entries(data.params ?? {}).slice(0, MAX_INLINE_PARAMS);
  const family = glyphFamily(data.category);
  const diameter = QUIET_CATEGORIES.has(data.category)
    ? QUIET_DIAMETER
    : (data.diameter ?? DEFAULT_DIAMETER);
  const sub = family === "block" ? blockSub(data.type) : null;
  // Inset handles to the glyph's left/right edge so they sit ON the shape, making
  // hand-wiring intuitive regardless of how small the glyph is.
  const glyphWidth = diameter * (WIDTH_FACTOR[family] ?? 1);
  const handleInset = Math.max(0, (NODE_BOX - glyphWidth) / 2);

  return (
    <div
      className={`nn-node${data.synthetic ? " nn-node--synthetic" : ""}`}
      title={label}
      style={{ "--node-color": color, "--d": `${diameter}px` }}
    >
      {data.type !== "input" && (
        <Handle type="target" position={Position.Left} className="nn-handle" style={{ left: handleInset }} />
      )}

      <div className={`nn-glyph nn-glyph--${family}${selected ? " is-selected" : ""}`}>
        {family === "stack" && (
          <span className="nn-glyph__dots" aria-hidden="true">
            <i /><i /><i /><i />
          </span>
        )}
        {family === "loop" && <span className="nn-glyph__loop" aria-hidden="true">↺</span>}
        {family === "merge" && <span className="nn-glyph__sign" aria-hidden="true">{mergeSign(data.type)}</span>}
        <span className="nn-node__label">{label}</span>
        {sub && <span className="nn-glyph__sub" aria-hidden="true">{sub}</span>}
      </div>

      {data.width != null && (
        <span className="nn-node__dim" aria-hidden="true">{data.width}</span>
      )}

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

      <Handle type="source" position={Position.Right} className="nn-handle" style={{ right: handleInset }} />
    </div>
  );
}
