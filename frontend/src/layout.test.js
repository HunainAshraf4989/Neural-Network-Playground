import { describe, it, expect } from "vitest";
import { computeLayout } from "./layout.js";

// Helpers to read column/row out of the {x,y} positions.
const COL_GAP = 240;
const ROW_GAP = 110;
const col = (p) => p.x / COL_GAP;
const row = (p) => p.y / ROW_GAP;

describe("computeLayout", () => {
  it("returns empty for no nodes", () => {
    expect(computeLayout({ nodes: [], edges: [] })).toEqual({});
    expect(computeLayout(undefined)).toEqual({});
  });

  it("places a lone input node at column 0, row 0", () => {
    const pos = computeLayout({ nodes: [{ id: "n1", type: "input" }], edges: [] });
    expect(pos.n1).toEqual({ x: 0, y: 0 });
  });

  it("assigns columns by BFS distance from input along a chain", () => {
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
    // input → n2 and input → n3 ; both at column 1, ordered n2 above n3.
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
    // n1→n2→n3→n5 (skip) and n1→n4→n5 ; n5 should be at column 4 (via the long
    // branch), never column 2 (via the short branch).
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

  it("puts nodes with no path from input at column 0 (still visible)", () => {
    const arch = {
      nodes: [
        { id: "n1", type: "input" },
        { id: "n2", type: "conv2d" }, // connected
        { id: "n3", type: "relu" }, // island, no edges
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
