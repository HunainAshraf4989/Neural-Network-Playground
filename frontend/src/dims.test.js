import { describe, it, expect } from "vitest";
import {
  ownWidth,
  computeWidths,
  computeDiameters,
  FALLBACK_WIDTH,
  MIN_DIAMETER,
  MAX_DIAMETER,
} from "./dims.js";

describe("ownWidth", () => {
  it("reads the feature count a layer emits from its own params", () => {
    expect(ownWidth({ type: "linear", params: { out_features: 128 } })).toBe(128);
    expect(ownWidth({ type: "conv2d", params: { out_channels: 32 } })).toBe(32);
    expect(ownWidth({ type: "input", params: { shape: [1, 28, 28] } })).toBe(784);
    expect(ownWidth({ type: "lstm", params: { hidden_size: 64, bidirectional: true } })).toBe(128);
    expect(ownWidth({ type: "embedding", params: { embedding_dim: 50 } })).toBe(50);
    expect(ownWidth({ type: "transformer_encoder_layer", params: { d_model: 256 } })).toBe(256);
  });

  it("returns null for pass-through layers (they inherit)", () => {
    expect(ownWidth({ type: "relu", params: {} })).toBeNull();
    expect(ownWidth({ type: "dropout", params: { p: 0.5 } })).toBeNull();
    expect(ownWidth({ type: "flatten", params: {} })).toBeNull();
    expect(ownWidth({ type: "maxpool2d", params: { kernel_size: 2 } })).toBeNull();
  });
});

describe("computeWidths", () => {
  it("inherits width through pass-through layers", () => {
    const arch = {
      nodes: [
        { id: "n1", type: "input", params: { shape: [1, 28, 28] } },
        { id: "n2", type: "flatten", params: {} },
        { id: "n3", type: "linear", params: { in_features: 784, out_features: 128 } },
        { id: "n4", type: "relu", params: {} },
      ],
      edges: [
        { from: "n1", to: "n2" },
        { from: "n2", to: "n3" },
        { from: "n3", to: "n4" },
      ],
    };
    const w = computeWidths(arch);
    expect(w.n1).toBe(784);
    expect(w.n2).toBe(784); // flatten inherits the input's size
    expect(w.n3).toBe(128);
    expect(w.n4).toBe(128); // relu inherits the linear's size
  });

  it("an autoencoder pinches to its latent (the minimum)", () => {
    const dims = [784, 128, 64, 32, 64, 128, 784];
    const nodes = [{ id: "n0", type: "input", params: { shape: [784] } }];
    const edges = [];
    dims.forEach((d, i) => {
      const id = `n${i + 1}`;
      nodes.push({ id, type: "linear", params: { out_features: d } });
      edges.push({ from: i === 0 ? "n0" : `n${i}`, to: id });
    });
    const w = computeWidths({ nodes, edges });
    expect(w.n4).toBe(32); // the latent
    expect(Math.min(...Object.values(w))).toBe(32);
  });

  it("concat sums its inputs; add takes the max", () => {
    const lin = (id, out) => ({ id, type: "linear", params: { out_features: out } });
    const arch = {
      nodes: [
        { id: "n1", type: "input", params: { shape: [10] } },
        lin("n2", 30),
        lin("n3", 50),
        { id: "n4", type: "concat", params: { dim: 1 } },
        { id: "n5", type: "add", params: {} },
      ],
      edges: [
        { from: "n1", to: "n2" }, { from: "n1", to: "n3" },
        { from: "n2", to: "n4" }, { from: "n3", to: "n4" },
        { from: "n2", to: "n5" }, { from: "n3", to: "n5" },
      ],
    };
    const w = computeWidths(arch);
    expect(w.n4).toBe(80); // 30 + 50
    expect(w.n5).toBe(50); // max(30, 50)
  });

  it("falls back for an orphan and never loops on a cycle", () => {
    expect(computeWidths({ nodes: [{ id: "n1", type: "relu", params: {} }], edges: [] }).n1)
      .toBe(FALLBACK_WIDTH);
    const cyc = computeWidths({
      nodes: [{ id: "a", type: "relu", params: {} }, { id: "b", type: "relu", params: {} }],
      edges: [{ from: "a", to: "b" }, { from: "b", to: "a" }],
    });
    expect(Number.isFinite(cyc.a)).toBe(true);
    expect(Number.isFinite(cyc.b)).toBe(true);
  });

  it("returns empty for an empty graph", () => {
    expect(computeWidths({ nodes: [], edges: [] })).toEqual({});
  });
});

describe("computeDiameters", () => {
  it("maps the widest layer to MAX and the narrowest to MIN", () => {
    const arch = {
      nodes: [
        { id: "n1", type: "linear", params: { out_features: 32 } },
        { id: "n2", type: "linear", params: { out_features: 784 } },
      ],
      edges: [{ from: "n1", to: "n2" }],
    };
    const d = computeDiameters(arch);
    expect(d.n1).toBe(MIN_DIAMETER);
    expect(d.n2).toBe(MAX_DIAMETER);
  });

  it("renders equal widths at the midpoint, not as a false bottleneck", () => {
    const arch = {
      nodes: [
        { id: "n1", type: "linear", params: { out_features: 128 } },
        { id: "n2", type: "linear", params: { out_features: 128 } },
      ],
      edges: [{ from: "n1", to: "n2" }],
    };
    const d = computeDiameters(arch);
    expect(d.n1).toBe(d.n2);
    expect(d.n1).toBe(Math.round((MIN_DIAMETER + MAX_DIAMETER) / 2));
  });

  it("returns empty for an empty graph", () => {
    expect(computeDiameters({ nodes: [], edges: [] })).toEqual({});
  });
});
