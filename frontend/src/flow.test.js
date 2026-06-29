import { describe, it, expect } from "vitest";
import { toFlowElements } from "./flow.js";

describe("toFlowElements", () => {
  it("maps an empty architecture to empty arrays", () => {
    expect(toFlowElements({ nodes: [], edges: [] })).toEqual({ nodes: [], edges: [] });
    expect(toFlowElements(undefined)).toEqual({ nodes: [], edges: [] });
  });

  it("maps nodes to the single 'layer' type with type/params/category in data", () => {
    const arch = {
      nodes: [{ id: "n1", type: "conv2d", params: { in_channels: 3 } }],
      edges: [],
    };
    const { nodes } = toFlowElements(arch);
    expect(nodes).toHaveLength(1);
    expect(nodes[0]).toMatchObject({
      id: "n1",
      type: "layer",
      data: { type: "conv2d", params: { in_channels: 3 }, category: "conv" },
    });
    expect(nodes[0].position).toHaveProperty("x");
    expect(nodes[0].position).toHaveProperty("y");
  });

  it("maps edges to source/target with a stable id", () => {
    const arch = {
      nodes: [
        { id: "n1", type: "input", params: {} },
        { id: "n2", type: "relu", params: {} },
      ],
      edges: [{ from: "n1", to: "n2" }],
    };
    const { edges } = toFlowElements(arch);
    expect(edges).toEqual([{ id: "n1->n2", source: "n1", target: "n2" }]);
  });

  it("leaves category undefined for an unknown type rather than throwing", () => {
    const arch = { nodes: [{ id: "n1", type: "mystery", params: {} }], edges: [] };
    const { nodes } = toFlowElements(arch);
    expect(nodes[0].data.category).toBeUndefined();
  });
});
