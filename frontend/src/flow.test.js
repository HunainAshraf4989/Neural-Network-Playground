import { describe, it, expect } from "vitest";
import { toFlowElements, dropRemoveChanges } from "./flow.js";

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

describe("dropRemoveChanges", () => {
  it("strips `remove` changes (backend owns existence)", () => {
    const changes = [
      { type: "position", id: "n1" },
      { type: "remove", id: "n2" },
      { type: "select", id: "n3", selected: true },
    ];
    expect(dropRemoveChanges(changes)).toEqual([
      { type: "position", id: "n1" },
      { type: "select", id: "n3", selected: true },
    ]);
  });

  it("lets geometry/selection changes through untouched", () => {
    const changes = [
      { type: "dimensions", id: "n1" },
      { type: "position", id: "n1" },
      { type: "select", id: "n1" },
    ];
    expect(dropRemoveChanges(changes)).toEqual(changes);
  });

  it("handles an all-remove batch and an empty batch", () => {
    expect(dropRemoveChanges([{ type: "remove", id: "e1" }, { type: "remove", id: "e2" }])).toEqual([]);
    expect(dropRemoveChanges([])).toEqual([]);
  });
});
