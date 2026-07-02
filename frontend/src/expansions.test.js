import { describe, it, expect } from "vitest";
import { expandGraph, expandNode, isExpandable, isUngroupable, EXPANDABLE_TYPES } from "./expansions.js";

const idsOf = (graph) => graph.nodes.map((n) => n.id);
const hasEdge = (graph, from, to) =>
  graph.edges.some((e) => e.from === from && e.to === to);

describe("expandGraph — pass-through", () => {
  it("leaves a graph with no composite layers untouched", () => {
    const arch = {
      nodes: [
        { id: "n1", type: "input", params: { shape: [1, 28, 28] } },
        { id: "n2", type: "conv2d", params: { out_channels: 16 } },
        { id: "n3", type: "relu", params: {} },
      ],
      edges: [{ from: "n1", to: "n2" }, { from: "n2", to: "n3" }],
    };
    const out = expandGraph(arch);
    expect(out.nodes).toEqual(arch.nodes);
    expect(out.edges).toEqual(arch.edges);
  });

  it("preserves the layout hint channel", () => {
    const arch = { nodes: [], edges: [], layout: { n1: { col: 2, row: 0 } } };
    expect(expandGraph(arch).layout).toEqual({ n1: { col: 2, row: 0 } });
  });

  it("tolerates empty / missing input", () => {
    expect(expandGraph(undefined)).toEqual({ nodes: [], edges: [], layout: undefined });
    expect(expandGraph({ nodes: [], edges: [] }).nodes).toEqual([]);
  });
});

describe("transformer_encoder_layer expansion", () => {
  const arch = {
    nodes: [
      { id: "n1", type: "input", params: { shape: [16, 128] } },
      { id: "n2", type: "transformer_encoder_layer", params: { d_model: 128, nhead: 8, dim_feedforward: 512, dropout: 0.1 } },
      { id: "n3", type: "linear", params: { in_features: 128, out_features: 10 } },
    ],
    edges: [{ from: "n1", to: "n2" }, { from: "n2", to: "n3" }],
  };
  const out = expandGraph(arch);

  it("replaces the one node with its 11 internal sub-nodes (real nodes kept)", () => {
    expect(idsOf(out)).toEqual(expect.arrayContaining([
      "n1", "n3",
      "n2/attn", "n2/do1", "n2/res1", "n2/norm1",
      "n2/fc1", "n2/act", "n2/do2", "n2/fc2", "n2/do3", "n2/res2", "n2/norm2",
    ]));
    expect(idsOf(out)).not.toContain("n2");
    expect(out.nodes).toHaveLength(2 + 11);
  });

  it("rewires the incoming edge into BOTH the attn and the attention residual", () => {
    expect(hasEdge(out, "n1", "n2/attn")).toBe(true);
    expect(hasEdge(out, "n1", "n2/res1")).toBe(true);
  });

  it("rewires the outgoing edge from the exit (norm2)", () => {
    expect(hasEdge(out, "n2/norm2", "n3")).toBe(true);
  });

  it("draws the feed-forward residual skip (norm1 -> res2)", () => {
    expect(hasEdge(out, "n2/norm1", "n2/res2")).toBe(true);
  });

  it("carries the right params onto sized sub-nodes", () => {
    const byId = Object.fromEntries(out.nodes.map((n) => [n.id, n]));
    expect(byId["n2/attn"].params).toMatchObject({ embed_dim: 128, num_heads: 8 });
    expect(byId["n2/fc1"].params).toMatchObject({ in_features: 128, out_features: 512 });
    expect(byId["n2/fc2"].params).toMatchObject({ in_features: 512, out_features: 128 });
  });

  it("flags every sub-node synthetic", () => {
    for (const n of out.nodes) {
      if (n.id.startsWith("n2/")) expect(n.synthetic).toBe(true);
    }
  });
});

describe("multihead_attention expansion", () => {
  const out = expandNode({ id: "n5", type: "multihead_attention", params: { embed_dim: 64, num_heads: 4, dropout: 0.0 } });

  it("is Q/K/V -> SDPA -> out", () => {
    expect(out.nodes.map((n) => n.id)).toEqual(["n5/q", "n5/k", "n5/v", "n5/sdpa", "n5/out"]);
    expect(out.entries).toEqual(["n5/q", "n5/k", "n5/v"]);
    expect(out.exit).toBe("n5/out");
  });

  it("gives the attention core an explicit category (no catalog type)", () => {
    const sdpa = out.nodes.find((n) => n.id === "n5/sdpa");
    expect(sdpa.category).toBe("attention");
    expect(sdpa.synthetic).toBe(true);
  });
});

describe("recurrent expansion", () => {
  it("does NOT expand a plain 1-layer unidirectional RNN", () => {
    expect(expandNode({ id: "n2", type: "rnn", params: { input_size: 64, hidden_size: 128, num_layers: 1, bidirectional: false } })).toBeNull();
    expect(isExpandable({ id: "n2", type: "rnn", params: { num_layers: 1, bidirectional: false } })).toBe(false);
  });

  it("unrolls a 2-layer LSTM into chained cells with the right input sizes", () => {
    const out = expandNode({ id: "n2", type: "lstm", params: { input_size: 64, hidden_size: 128, num_layers: 2, bidirectional: false } });
    const byId = Object.fromEntries(out.nodes.map((n) => [n.id, n]));
    expect(byId["n2/l0"].params.input_size).toBe(64);
    expect(byId["n2/l1"].params.input_size).toBe(128); // fed by layer-0's hidden state
    expect(out.edges).toContainEqual({ from: "n2/l0", to: "n2/l1" });
    expect(out.entries).toEqual(["n2/l0"]);
    expect(out.exit).toBe("n2/l1");
  });

  it("splits a bidirectional layer into forward + backward -> concat", () => {
    const out = expandNode({ id: "n2", type: "gru", params: { input_size: 64, hidden_size: 128, num_layers: 1, bidirectional: true } });
    expect(out.nodes.map((n) => n.id)).toEqual(["n2/l0f", "n2/l0b", "n2/l0c"]);
    expect(out.edges).toContainEqual({ from: "n2/l0f", to: "n2/l0c" });
    expect(out.edges).toContainEqual({ from: "n2/l0b", to: "n2/l0c" });
    expect(out.exit).toBe("n2/l0c");
  });
});

describe("edges between two expandable nodes", () => {
  it("rewires exit-of-A into entries-of-B", () => {
    const arch = {
      nodes: [
        { id: "n1", type: "multihead_attention", params: { embed_dim: 32, num_heads: 4, dropout: 0 } },
        { id: "n2", type: "multihead_attention", params: { embed_dim: 32, num_heads: 4, dropout: 0 } },
      ],
      edges: [{ from: "n1", to: "n2" }],
    };
    const out = expandGraph(arch);
    // n1's out proj feeds each of n2's Q/K/V projections.
    expect(hasEdge(out, "n1/out", "n2/q")).toBe(true);
    expect(hasEdge(out, "n1/out", "n2/k")).toBe(true);
    expect(hasEdge(out, "n1/out", "n2/v")).toBe(true);
  });
});

describe("isUngroupable", () => {
  it("allows the transformer block and stacked/bidirectional recurrent cells", () => {
    expect(isUngroupable({ type: "transformer_encoder_layer", params: { d_model: 32, nhead: 4 } })).toBe(true);
    expect(isUngroupable({ type: "lstm", params: { num_layers: 2, bidirectional: false } })).toBe(true);
    expect(isUngroupable({ type: "gru", params: { num_layers: 1, bidirectional: true } })).toBe(true);
  });

  it("excludes plain multihead_attention (needs the synthetic SDPA type)", () => {
    expect(isUngroupable({ type: "multihead_attention", params: { embed_dim: 32, num_heads: 4 } })).toBe(false);
  });

  it("excludes 1-layer unidirectional recurrent cells and everything else", () => {
    expect(isUngroupable({ type: "rnn", params: { num_layers: 1, bidirectional: false } })).toBe(false);
    expect(isUngroupable({ type: "relu", params: {} })).toBe(false);
    expect(isUngroupable(null)).toBe(false);
  });
});

describe("EXPANDABLE_TYPES", () => {
  it("lists exactly the composite families", () => {
    expect([...EXPANDABLE_TYPES].sort()).toEqual(
      ["gru", "lstm", "multihead_attention", "rnn", "transformer_encoder_layer"].sort(),
    );
  });
});
