// One node per layer (spec §13), but its GLYPH is chosen by category so the graph
// reads like the conventional diagram for whatever was built - a CNN's conv slabs,
// an MLP's neuron stacks, a transformer's blocks - and its SIZE scales with the
// layer's feature width (see dims.js) so the network's silhouette is legible: an
// autoencoder visibly pinches at its latent, a CNN's channels swell. The idiom is
// derived from the graph, never toggled.
//
// Invariants kept across all glyphs: the `input` type has no target handle, every
// other type has one target + one source handle (React Flow allows many edges into
// one target handle, so merge nodes need no special wiring); the type name shows on
// the glyph and the key params live in a hover tooltip (full set in the panel).

import { Fragment } from "react";
import { Handle, Position } from "@xyflow/react";
import { colorOfCategory } from "./catalog.js";
import { NEURON_CIRCLE, neuronColumnHalfHeight, neuronSplit } from "./dims.js";

// Categories that, in the EXPANDED view, render as a column of neuron circles (the
// classic deep-net diagram) rather than an iconic glyph: the input features, the
// fully-connected layers, and the per-neuron activations applied to them.
export const NEURON_CATEGORIES = new Set(["input", "linear", "activation"]);

const MAX_INLINE_PARAMS = 4;
const DEFAULT_DIAMETER = 92; // used when a node is rendered without a computed size

// In-place ops (activation/norm/dropout/flatten/pooling) operate on whatever
// tensor flows through them - they aren't representations, so they render at a
// fixed small size and stay "quiet", letting the structural layers (linear/conv/…)
// carry the silhouette instead of a giant relu dwarfing its own linear.
const QUIET_CATEGORIES = new Set(["activation", "norm", "regularization", "shape", "pooling"]);
const QUIET_DIAMETER = 64;

// The node sits in a fixed-size cell (matches .nn-node in styles.css) with the
// glyph centered inside, so connection handles must be INSET to the glyph's real
// edge - otherwise they float in the empty cell, away from the visible shape, and
// the graph becomes fiddly to wire by hand. Each family's width relative to `--d`:
const NODE_BOX = 150;
const WIDTH_FACTOR = { stack: 0.72, block: 1.18, loop: 1.18, merge: 0.82, tag: 1.12 };

function formatValue(v) {
  if (v === null) return "auto";
  if (Array.isArray(v)) return `[${v.join(",")}]`;
  return String(v);
}

// Break long, underscore-joined type names (e.g. transformer_encoder_layer) at
// their word boundaries so a label wraps BY WORD - never mid-letter - and stays
// inside the glyph. A `<wbr>` after each underscore/space is a *soft* break the
// browser only takes when the line overflows (paired with overflow-wrap in the
// .nn-node__label CSS, which only splits a single word as a last resort).
function labelWithBreaks(label) {
  return String(label)
    .split(/([_ ])/) // keep the separators as their own chunks
    .map((chunk, i) =>
      chunk === "_" || chunk === " "
        ? <Fragment key={i}>{chunk}<wbr /></Fragment>
        : <Fragment key={i}>{chunk}</Fragment>,
    );
}

// Glyph family per category - the visual vocabulary that makes each NN family
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

// The EXPANDED view of a per-neuron layer (input / linear / activation): a column of
// large solid neuron circles, the classic deep-net diagram. `count` (from
// dims.computeNeuronCounts) is a representative number scaled to the layer's size, so
// a wider layer shows more circles; the exact size stays in the dim label, and the
// fully-connected lines between columns are drawn by NeuronBundleEdge. Collapsed,
// these layers keep their iconic glyph - the neurons only appear once you Expand.
//
// When the layer truly has more neurons than we draw (`truncated`), a vertical ⋮ sits
// in the middle to stand for the omitted units - the classic "…and N more neurons"
// cue. It occupies one circle-slot (see dims.neuronSplit / neuronYOffsets), so the
// bundle edges still land on the visible circles.
function NeuronColumn({ count, truncated }) {
  const n = count && count > 0 ? count : 3;
  const { top, bottom } = neuronSplit(n, truncated);
  return (
    <span className="nn-neuron-col" aria-hidden="true">
      {Array.from({ length: top }, (_, i) => <i key={`t${i}`} />)}
      {truncated && (
        <span className="nn-neuron-col__more" aria-hidden="true">
          <b /><b /><b />
        </span>
      )}
      {Array.from({ length: bottom }, (_, i) => <i key={`b${i}`} />)}
    </span>
  );
}

// Display-only short names. A glyph's SIZE encodes feature width (dims.js), so a
// layer whose width is the graph's minimum renders at MIN_DIAMETER - about 71x50px
// for a block. The raw schema type `transformer_encoder_layer` cannot fit there at a
// readable size: its longest word alone overflows the shell, so the CSS last-resort
// break splits it mid-letter and the extra lines spill outside the border. These are
// the names a real architecture diagram would use anyway; the exact type is still in
// the hover tooltip, the params panel, and the generated code. Same idea as the
// friendlier `data.label` carried by expansion sub-nodes.
const DISPLAY_NAMES = {
  transformer_encoder_layer: "encoder layer",
  positional_encoding: "pos. encoding",
  multihead_attention: "attention",
  adaptive_avg_pool2d: "adaptive avgpool",
  adaptive_max_pool2d: "adaptive maxpool",
  conv_transpose2d: "conv transpose",
  instancenorm2d: "instance norm",
  groupnorm: "group norm",
  batchnorm1d: "batch norm 1d",
  batchnorm2d: "batch norm 2d",
  layernorm: "layer norm",
  leaky_relu: "leaky relu",
};

function displayName(label) {
  return DISPLAY_NAMES[label] ?? label;
}

function blockSub(type) {
  if (type === "multihead_attention") return "MHA";
  if (type === "transformer_encoder_layer") return "MHA·FFN";
  if (type === "positional_encoding") return "POS";
  return null;
}

export default function LayerNode({ data, selected }) {
  // Color from category (not type) so expansion sub-nodes with a synthetic type
  // (e.g. the attention core) still color correctly; for a normal node the
  // category IS the type's category, so this is unchanged.
  const color = colorOfCategory(data.category);
  const label = data.label ?? data.type; // sub-nodes carry a friendlier label
  const shown = displayName(label); // shortened to fit the glyph; `title` keeps the real type
  const params = Object.entries(data.params ?? {}).slice(0, MAX_INLINE_PARAMS);
  const family = glyphFamily(data.category);
  // In the EXPANDED view, per-neuron layers (input / linear / activation) render as a
  // column of neuron circles - the classic deep-net diagram. Collapsed, or for the
  // synthetic sub-layers inside an expanded composite block, they keep their glyph.
  const showNeurons = data.expanded && !data.synthetic && NEURON_CATEGORIES.has(data.category);
  const diameter = QUIET_CATEGORIES.has(data.category)
    ? QUIET_DIAMETER
    : (data.diameter ?? DEFAULT_DIAMETER);
  const sub = family === "block" ? blockSub(data.type) : null;
  const neuronCount = data.neurons && data.neurons > 0 ? data.neurons : 3;
  // Show the ⋮ only when the layer's real width genuinely exceeds the circles drawn,
  // so a small layer (e.g. a 3-unit output) renders its exact neurons with no ellipsis.
  const neuronsTruncated = showNeurons && data.width != null && data.width > neuronCount;
  // Inset handles to the glyph's left/right edge so they sit ON the shape, making
  // hand-wiring intuitive; for a neuron column the "shape" is the circle stack.
  const glyphWidth = showNeurons ? NEURON_CIRCLE : diameter * (WIDTH_FACTOR[family] ?? 1);
  const handleInset = Math.max(0, (NODE_BOX - glyphWidth) / 2);
  const style = { "--node-color": color, "--d": `${diameter}px` };
  // Column half-height positions the labels just below the circles (see CSS), since a
  // neuron column can be taller than the node cell.
  if (showNeurons) style["--col-h"] = `${neuronColumnHalfHeight(neuronCount, neuronsTruncated)}px`;

  return (
    <div
      className={`nn-node${data.synthetic ? " nn-node--synthetic" : ""}${showNeurons ? " nn-node--neurons" : ""}`}
      title={label}
      style={style}
    >
      {data.type !== "input" && (
        <Handle type="target" position={Position.Left} className="nn-handle" style={{ left: handleInset }} />
      )}

      {showNeurons ? (
        <>
          <NeuronColumn count={neuronCount} truncated={neuronsTruncated} />
          <span className="nn-node__label">{labelWithBreaks(shown)}</span>
        </>
      ) : (
        <div className={`nn-glyph nn-glyph--${family}${selected ? " is-selected" : ""}`}>
          {family === "stack" && (
            <span className="nn-glyph__dots" aria-hidden="true">
              <i /><i /><i /><i />
            </span>
          )}
          {family === "loop" && <span className="nn-glyph__loop" aria-hidden="true">↺</span>}
          {family === "merge" && <span className="nn-glyph__sign" aria-hidden="true">{mergeSign(data.type)}</span>}
          <span className="nn-node__label">{labelWithBreaks(shown)}</span>
          {sub && <span className="nn-glyph__sub" aria-hidden="true">{sub}</span>}
        </div>
      )}

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
