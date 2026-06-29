import { describe, it, expect } from "vitest";
import { paramKind, toRaw, coerce } from "./paramTypes.js";

describe("paramKind", () => {
  it("classifies by the catalog default value", () => {
    expect(paramKind(true)).toBe("boolean");
    expect(paramKind(false)).toBe("boolean");
    expect(paramKind([1, 2, 3])).toBe("array");
    expect(paramKind(null)).toBe("nullable");
    expect(paramKind(3)).toBe("number");
    expect(paramKind(0.5)).toBe("number");
    expect(paramKind("float32")).toBe("string");
  });
});

describe("toRaw", () => {
  it("keeps booleans as booleans", () => {
    expect(toRaw(true)).toBe(true);
    expect(toRaw(false)).toBe(false);
  });
  it("comma-joins arrays", () => {
    expect(toRaw([1, 28, 28])).toBe("1,28,28");
  });
  it("renders null/undefined as empty string", () => {
    expect(toRaw(null)).toBe("");
    expect(toRaw(undefined)).toBe("");
  });
  it("stringifies numbers and strings", () => {
    expect(toRaw(3)).toBe("3");
    expect(toRaw(0.1)).toBe("0.1");
    expect(toRaw("nearest")).toBe("nearest");
  });
});

describe("coerce", () => {
  it("boolean ← truthiness", () => {
    expect(coerce("boolean", true)).toBe(true);
    expect(coerce("boolean", false)).toBe(false);
  });

  it("number ← numeric string, round-trips integers and floats", () => {
    expect(coerce("number", "3")).toBe(3);
    expect(coerce("number", "0.5")).toBe(0.5);
    expect(coerce("number", "-1")).toBe(-1);
  });

  it("number keeps a non-numeric string as-is (surfaces at validate time)", () => {
    expect(coerce("number", "abc")).toBe("abc");
  });

  it("nullable: empty → null, value → number", () => {
    expect(coerce("nullable", "")).toBe(null);
    expect(coerce("nullable", "  ")).toBe(null);
    expect(coerce("nullable", "2")).toBe(2);
  });

  it("array: comma list of numbers", () => {
    expect(coerce("array", "1,28,28")).toEqual([1, 28, 28]);
    expect(coerce("array", "1, 2 , 3")).toEqual([1, 2, 3]);
  });

  it("array: JSON list", () => {
    expect(coerce("array", "[1,2,3]")).toEqual([1, 2, 3]);
  });

  it("array: empty → []", () => {
    expect(coerce("array", "")).toEqual([]);
  });

  it("array: non-numeric items stay strings", () => {
    expect(coerce("array", "nearest,linear")).toEqual(["nearest", "linear"]);
  });

  it("string passthrough", () => {
    expect(coerce("string", "float32")).toBe("float32");
    expect(coerce("string", "")).toBe("");
    expect(coerce("string", null)).toBe("");
  });

  it("round-trips a representative param set through toRaw → coerce", () => {
    const cases = [
      ["number", 3],
      ["number", 0.1],
      ["array", [1, 28, 28]],
      ["nullable", null],
      ["string", "float32"],
      ["boolean", true],
    ];
    for (const [kind, value] of cases) {
      expect(coerce(kind, toRaw(value))).toEqual(value);
    }
  });
});
