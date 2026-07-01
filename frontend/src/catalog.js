// Frontend mirror of the backend layer catalog (backend/layers.py CATALOG).
//
// The backend rejects add_layer unless every *required* param is present, but
// required params have no schema default there. So this catalog carries a
// concrete starting value for EVERY param (required + optional). Dragging a
// palette item sends add_layer with these values; the user then refines them in
// the params panel. Optional defaults match the backend's exactly.
//
// `catalog.test.js` asserts this stays in lockstep with backend/layers.py:
// same types, same categories, and params === required ∪ optional for each
// type (so a drag can never be rejected for a missing or unknown param).

export const CATALOG = {
  // --- input ---
  input: { category: "input", params: { shape: [1, 28, 28], dtype: "float32" } },

  // --- conv ---
  conv2d: {
    category: "conv",
    params: { in_channels: 1, out_channels: 16, kernel_size: 3, stride: 1, padding: 0, dilation: 1 },
  },
  conv1d: {
    category: "conv",
    params: { in_channels: 1, out_channels: 16, kernel_size: 3, stride: 1, padding: 0, dilation: 1 },
  },
  conv3d: {
    category: "conv",
    params: { in_channels: 1, out_channels: 16, kernel_size: 3, stride: 1, padding: 0, dilation: 1 },
  },
  conv_transpose2d: {
    category: "conv",
    params: {
      in_channels: 16, out_channels: 8, kernel_size: 2,
      stride: 2, padding: 0, output_padding: 0, dilation: 1,
    },
  },

  // --- pooling / resampling ---
  maxpool2d: { category: "pooling", params: { kernel_size: 2, stride: null, padding: 0 } },
  avgpool2d: { category: "pooling", params: { kernel_size: 2, stride: null, padding: 0 } },
  maxpool1d: { category: "pooling", params: { kernel_size: 2, stride: null, padding: 0 } },
  avgpool1d: { category: "pooling", params: { kernel_size: 2, stride: null, padding: 0 } },
  adaptive_avg_pool2d: { category: "pooling", params: { output_size: [1, 1] } },
  adaptive_max_pool2d: { category: "pooling", params: { output_size: [1, 1] } },
  upsample: { category: "pooling", params: { scale_factor: 2, mode: "nearest" } },

  // --- norm ---
  batchnorm2d: { category: "norm", params: { num_features: 16 } },
  batchnorm1d: { category: "norm", params: { num_features: 16 } },
  groupnorm: { category: "norm", params: { num_groups: 4, num_channels: 16 } },
  instancenorm2d: { category: "norm", params: { num_features: 16 } },
  layernorm: { category: "norm", params: { normalized_shape: 128 } },

  // --- activation ---
  relu: { category: "activation", params: {} },
  gelu: { category: "activation", params: {} },
  sigmoid: { category: "activation", params: {} },
  tanh: { category: "activation", params: {} },
  leaky_relu: { category: "activation", params: { negative_slope: 0.01 } },
  elu: { category: "activation", params: { alpha: 1.0 } },
  softmax: { category: "activation", params: { dim: -1 } },

  // --- shape / dense / regularization / embedding ---
  flatten: { category: "shape", params: { start_dim: 1 } },
  linear: { category: "linear", params: { in_features: 64, out_features: 10 } },
  dropout: { category: "regularization", params: { p: 0.5 } },
  embedding: { category: "embedding", params: { num_embeddings: 1000, embedding_dim: 64 } },

  // --- recurrent ---
  rnn: { category: "recurrent", params: { input_size: 64, hidden_size: 128, num_layers: 1, bidirectional: false } },
  lstm: { category: "recurrent", params: { input_size: 64, hidden_size: 128, num_layers: 1, bidirectional: false } },
  gru: { category: "recurrent", params: { input_size: 64, hidden_size: 128, num_layers: 1, bidirectional: false } },

  // --- attention / transformer ---
  positional_encoding: { category: "attention", params: { d_model: 128, max_len: 5000 } },
  multihead_attention: { category: "attention", params: { embed_dim: 128, num_heads: 8, dropout: 0.0 } },
  transformer_encoder_layer: {
    category: "attention",
    params: { d_model: 128, nhead: 8, dim_feedforward: 2048, dropout: 0.1 },
  },

  // --- merge (forward-only, no nn.Module) ---
  add: { category: "merge", params: {} },
  concat: { category: "merge", params: { dim: 1 } },
};

// Per-category color (frontend-only). Used by the palette and node borders.
export const CATEGORY_COLORS = {
  input: "#5b8def",
  conv: "#e8743b",
  pooling: "#19a979",
  norm: "#945ecf",
  activation: "#ed4192",
  recurrent: "#13a4b4",
  attention: "#f6c85f",
  merge: "#6f4e7c",
  linear: "#0b84a5",
  shape: "#8c8c8c",
  regularization: "#ca472f",
  embedding: "#c85200",
};

export const FALLBACK_COLOR = "#8c8c8c";

export function categoryOf(type) {
  return CATALOG[type]?.category;
}

export function colorOfCategory(category) {
  return CATEGORY_COLORS[category] ?? FALLBACK_COLOR;
}

export function colorOf(type) {
  return colorOfCategory(categoryOf(type));
}

// Fresh deep copy of a type's default params, ready to send in add_layer.
export function defaultParams(type) {
  return structuredClone(CATALOG[type].params);
}

// Palette grouping: [{ category, color, types: [...] }], in catalog order.
export function groupedByCategory() {
  const groups = new Map();
  for (const [type, entry] of Object.entries(CATALOG)) {
    if (!groups.has(entry.category)) {
      groups.set(entry.category, {
        category: entry.category,
        color: CATEGORY_COLORS[entry.category] ?? FALLBACK_COLOR,
        types: [],
      });
    }
    groups.get(entry.category).types.push(type);
  }
  return [...groups.values()];
}
