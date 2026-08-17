import { describe, it, expect } from "vitest";
import { isSkipSpan, skipArc } from "./edgeGeometry.js";
import { COL_GAP } from "./layout.js";

describe("isSkipSpan", () => {
  it("treats an adjacent-column edge as a normal (non-skip) edge", () => {
    // adjacent handles are about (COL_GAP - cell width) apart - under one column
    expect(isSkipSpan(0, COL_GAP - 150)).toBe(false);
  });

  it("treats an edge that jumps over a layer as a skip", () => {
    expect(isSkipSpan(0, 2 * COL_GAP - 150)).toBe(true);
  });

  it("never treats a backward/zero span as a skip", () => {
    expect(isSkipSpan(300, 100)).toBe(false);
    expect(isSkipSpan(100, 100)).toBe(false);
  });
});

describe("skipArc", () => {
  it("starts at the source and ends at the target", () => {
    const { path } = skipArc(0, 100, 400, 120);
    expect(path.startsWith("M0,100")).toBe(true);
    expect(path.endsWith("400,120")).toBe(true);
  });

  it("bows above the higher of the two endpoints", () => {
    const { labelY } = skipArc(0, 100, 400, 120);
    expect(labelY).toBeLessThan(100);
  });

  it("lifts longer skips higher (but clamps)", () => {
    const short = skipArc(0, 0, 200, 0).labelY;
    const long = skipArc(0, 0, 1200, 0).labelY;
    expect(long).toBeLessThan(short); // longer arc → higher apex (smaller y)
  });
});
