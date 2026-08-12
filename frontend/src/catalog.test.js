import { describe, it, expect } from "vitest";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { CATALOG, CATEGORY_COLORS } from "./catalog.js";

// Ground truth: import backend/layers.py and dump its CATALOG. This guarantees
// the frontend mirror can never silently drift from the backend — if a layer is
// added, renamed, or its params change in layers.py, this test goes red until
// catalog.js matches, so palette drags can't start failing in production.
const here = path.dirname(fileURLToPath(import.meta.url));
const backendDir = path.resolve(here, "../../backend");
// The project venv (see CLAUDE.md); override with PYTHON on another machine.
const PYTHON = process.env.PYTHON || "/home/hunain/base/bin/python";

function pythonWorks() {
  try {
    execFileSync(PYTHON, ["--version"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function loadBackendCatalog() {
  const code = [
    "import json, layers",
    "out = {t: {'category': s['category'], 'params': sorted(list(s['required']) + list(s['optional']))}",
    "       for t, s in layers.CATALOG.items()}",
    "print(json.dumps(out))",
  ].join("\n");
  const stdout = execFileSync(PYTHON, ["-c", code], { cwd: backendDir });
  return JSON.parse(stdout.toString());
}

const backendAvailable = existsSync(path.join(backendDir, "layers.py")) && pythonWorks();
const maybe = backendAvailable ? describe : describe.skip;

maybe("catalog parity with backend/layers.py", () => {
  const backend = loadBackendCatalog();

  it("covers exactly the same layer types", () => {
    expect(Object.keys(CATALOG).sort()).toEqual(Object.keys(backend).sort());
  });

  it("matches category for every type", () => {
    for (const [type, spec] of Object.entries(backend)) {
      expect(CATALOG[type]?.category, `category for ${type}`).toBe(spec.category);
    }
  });

  it("supplies a value for every required+optional param and no extras", () => {
    // params === required ∪ optional means a palette drag can never be rejected
    // for a missing required param nor an unknown param.
    for (const [type, spec] of Object.entries(backend)) {
      const got = Object.keys(CATALOG[type].params).sort();
      expect(got, `params for ${type}`).toEqual(spec.params);
    }
  });

  it("has a color for every category present in the catalog", () => {
    const categories = new Set(Object.values(CATALOG).map((e) => e.category));
    for (const c of categories) {
      expect(CATEGORY_COLORS[c], `color for category ${c}`).toBeTruthy();
    }
  });
});

describe("catalog (always-on sanity)", () => {
  it("every entry has a category and a params object", () => {
    for (const [type, entry] of Object.entries(CATALOG)) {
      expect(typeof entry.category, type).toBe("string");
      expect(typeof entry.params, type).toBe("object");
    }
  });
});
