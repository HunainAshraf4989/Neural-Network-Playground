import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

// The canvas port has ONE source of truth: DEFAULT_WS_PORT in backend/config.py.
// vite.config.js parses it and injects `__NN_WS_PORT__`, which the websocket hook
// builds its default URL from. If that wiring ever breaks - the constant renamed,
// the regex stopped matching, the define dropped - the UI silently goes back to
// knocking on 8765 while the backend listens somewhere else, and the canvas just
// says "Waiting for the backend…" forever. This test is what makes that loud.
const here = path.dirname(fileURLToPath(import.meta.url));
const configPath = path.resolve(here, "../../backend/config.py");

describe("canvas port wiring", () => {
  it("injects the port declared in backend/config.py", () => {
    const source = readFileSync(configPath, "utf8");
    const match = source.match(/^DEFAULT_WS_PORT\s*=\s*(\d+)/m);
    expect(match, "DEFAULT_WS_PORT not found in backend/config.py").not.toBeNull();

    expect(typeof __NN_WS_PORT__).toBe("number");
    expect(__NN_WS_PORT__).toBe(Number(match[1]));
  });
});
