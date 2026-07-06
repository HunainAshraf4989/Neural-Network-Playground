// Deploy S1: pure helpers of the socket hook — backend-URL derivation and
// reconnect backoff. The hook's socket behavior itself is exercised end-to-end
// by test/e2e/ws.test.js against the real backend.

import { describe, it, expect } from "vitest";

import { wsUrl, healthzUrl, reconnectDelay } from "./useArchitectureSocket.js";

describe("backend URL derivation (single VITE_BACKEND_URL)", () => {
  it("derives ws:// from http:// and wss:// from https://", () => {
    expect(wsUrl("http://localhost:8765")).toBe("ws://localhost:8765/ws");
    expect(wsUrl("https://me-nnp.hf.space")).toBe("wss://me-nnp.hf.space/ws");
  });

  it("tolerates trailing slashes", () => {
    expect(wsUrl("https://me-nnp.hf.space/")).toBe("wss://me-nnp.hf.space/ws");
    expect(healthzUrl("https://me-nnp.hf.space/")).toBe("https://me-nnp.hf.space/healthz");
  });

  it("derives the healthz probe URL on the same origin", () => {
    expect(healthzUrl("http://localhost:8765")).toBe("http://localhost:8765/healthz");
  });
});

describe("reconnect backoff", () => {
  it("starts at the 1.5 s base (with jitter down to half)", () => {
    expect(reconnectDelay(0, 1)).toBe(1500);
    expect(reconnectDelay(0, 0)).toBe(750);
  });

  it("doubles per attempt and caps at 15 s", () => {
    expect(reconnectDelay(1, 1)).toBe(3000);
    expect(reconnectDelay(2, 1)).toBe(6000);
    expect(reconnectDelay(10, 1)).toBe(15000); // capped
  });

  it("jitters within [half, full] so tabs don't reconnect in lockstep", () => {
    for (let i = 0; i < 20; i++) {
      const d = reconnectDelay(3);
      expect(d).toBeGreaterThanOrEqual(6000);
      expect(d).toBeLessThanOrEqual(12000);
    }
  });
});
