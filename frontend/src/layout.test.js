import { describe, it, expect } from "vitest";
import {
  computeLayout,
  reachableFromInput,
  snapToCell,
  colOf,
  rowOf,
  COL_GAP,
  ROW_GAP,
} from "./layout.js";

const col = (p) => p.x / COL_GAP;
const row = (p) => p.y / ROW_GAP;
const cell = (c, r) => ({ x: c * COL_GAP, y: r * ROW_GAP });

describe("computeLayout", () => {
  it("returns empty for no nodes", () => {
    expect(computeLayout({ nodes: [], edges: [] })).toEqual({});
    expect(computeLayout(undefined)).toEqual({});
  });

  it("places a lone input node at column 0, row 0", () => {
    const pos = computeLayout({ nodes: [{ id: "n1", type: "input" }], edges: [] });
    expect(pos.n1).toEqual({ x: 0, y: 0 });
  });

  it("connecting n1->n2 puts n2 one column to the right of n1 (edge-driven)", () => {
    const arch = {
      nodes: [
        { id: "n1", type: "input" },
        { id: "n2", type: "conv2d" },
        { id: "n3", type: "relu" },
      ],
      edges: [
        { from: "n1", to: "n2" },
        { from: "n2", to: "n3" },
      ],
    };
    const pos = computeLayout(arch);
    expect(col(pos.n1)).toBe(0);
    expect(col(pos.n2)).toBe(1);
    expect(col(pos.n3)).toBe(2);
  });

  it("stacks nodes sharing a column by node-id order", () => {
    const arch = {
      nodes: [
        { id: "n1", type: "input" },
        { id: "n2", type: "conv2d" },
        { id: "n3", type: "conv2d" },
      ],
      edges: [
        { from: "n1", to: "n2" },
        { from: "n1", to: "n3" },
      ],
    };
    const pos = computeLayout(arch);
    expect(col(pos.n2)).toBe(1);
    expect(col(pos.n3)).toBe(1);
    expect(row(pos.n2)).toBe(0);
    expect(row(pos.n3)).toBe(1);
  });

  it("uses MAX distance so a merge sits right of its deepest predecessor", () => {
    const arch = {
      nodes: [
        { id: "n1", type: "input" },
        { id: "n2", type: "conv2d" },
        { id: "n3", type: "conv2d" },
        { id: "n4", type: "conv2d" },
        { id: "n5", type: "add" },
      ],
      edges: [
        { from: "n1", to: "n2" },
        { from: "n2", to: "n3" },
        { from: "n3", to: "n5" },
        { from: "n1", to: "n4" },
        { from: "n4", to: "n5" },
      ],
    };
    const pos = computeLayout(arch);
    expect(col(pos.n5)).toBe(3);
  });

  it("puts a not-yet-wired node at column 0 (still visible, then flows right once wired)", () => {
    const arch = {
      nodes: [
        { id: "n1", type: "input" },
        { id: "n2", type: "conv2d" }, // connected
        { id: "n3", type: "relu" }, // island, no edges yet
      ],
      edges: [{ from: "n1", to: "n2" }],
    };
    const pos = computeLayout(arch);
    expect(col(pos.n2)).toBe(1);
    expect(col(pos.n3)).toBe(0);
  });

  it("handles a graph with no input node (all at column 0, id order)", () => {
    const arch = {
      nodes: [
        { id: "n1", type: "relu" },
        { id: "n2", type: "relu" },
      ],
      edges: [{ from: "n1", to: "n2" }],
    };
    const pos = computeLayout(arch);
    expect(col(pos.n1)).toBe(0);
    expect(col(pos.n2)).toBe(0);
    expect(row(pos.n1)).toBe(0);
    expect(row(pos.n2)).toBe(1);
  });

  it("ignores edges referencing unknown nodes", () => {
    const arch = {
      nodes: [{ id: "n1", type: "input" }],
      edges: [{ from: "n1", to: "ghost" }],
    };
    const pos = computeLayout(arch);
    expect(pos.n1).toEqual({ x: 0, y: 0 });
    expect(pos.ghost).toBeUndefined();
  });

  it("is deterministic across runs", () => {
    const arch = {
      nodes: [
        { id: "n1", type: "input" },
        { id: "n2", type: "conv2d" },
        { id: "n3", type: "relu" },
      ],
      edges: [
        { from: "n1", to: "n2" },
        { from: "n2", to: "n3" },
      ],
    };
    expect(computeLayout(arch)).toEqual(computeLayout(arch));
  });
});

describe("reachableFromInput", () => {
  const set = (arch) => reachableFromInput(arch);

  it("is empty when there is no input node", () => {
    const arch = {
      nodes: [
        { id: "n1", type: "relu" },
        { id: "n2", type: "relu" },
      ],
      edges: [{ from: "n1", to: "n2" }],
    };
    expect(set(arch)).toEqual(new Set());
  });

  it("includes the input node itself", () => {
    expect(set({ nodes: [{ id: "n1", type: "input" }], edges: [] })).toEqual(new Set(["n1"]));
  });

  it("follows the whole chain from input", () => {
    const arch = {
      nodes: [
        { id: "n1", type: "input" },
        { id: "n2", type: "conv2d" },
        { id: "n3", type: "relu" },
      ],
      edges: [
        { from: "n1", to: "n2" },
        { from: "n2", to: "n3" },
      ],
    };
    expect(set(arch)).toEqual(new Set(["n1", "n2", "n3"]));
  });

  it("excludes a subgraph cut off from input (the edge-delete case)", () => {
    // input->n2 still wired; n3->n4 used to hang off n2 but that edge is gone.
    const arch = {
      nodes: [
        { id: "n1", type: "input" },
        { id: "n2", type: "conv2d" },
        { id: "n3", type: "relu" },
        { id: "n4", type: "linear" },
      ],
      edges: [
        { from: "n1", to: "n2" },
        { from: "n3", to: "n4" },
      ],
    };
    expect(set(arch)).toEqual(new Set(["n1", "n2"]));
  });

  it("reaches both arms of a branch+merge", () => {
    const arch = {
      nodes: [
        { id: "n1", type: "input" },
        { id: "n2", type: "conv2d" },
        { id: "n3", type: "conv2d" },
        { id: "n4", type: "add" },
      ],
      edges: [
        { from: "n1", to: "n2" },
        { from: "n1", to: "n3" },
        { from: "n2", to: "n4" },
        { from: "n3", to: "n4" },
      ],
    };
    expect(set(arch)).toEqual(new Set(["n1", "n2", "n3", "n4"]));
  });

  it("ignores edges referencing unknown nodes and handles empty input", () => {
    expect(set({ nodes: [{ id: "n1", type: "input" }], edges: [{ from: "n1", to: "ghost" }] }))
      .toEqual(new Set(["n1"]));
    expect(set({ nodes: [], edges: [] })).toEqual(new Set());
    expect(set(undefined)).toEqual(new Set());
  });
});

describe("grid helpers", () => {
  it("snapToCell snaps to the nearest cell and clamps to col/row >= 0", () => {
    expect(snapToCell({ x: COL_GAP * 2 + 5, y: ROW_GAP * 3 - 4 })).toEqual(cell(2, 3));
    expect(snapToCell({ x: -50, y: -999 })).toEqual(cell(0, 0));
  });

  it("colOf / rowOf recover the column and row from a position", () => {
    expect(colOf(cell(4, 2))).toBe(4);
    expect(rowOf(cell(4, 2))).toBe(2);
  });
});
