// End-to-end protocol test: boots the REAL standalone backend and drives the
// websocket using the frontend's own message builders (src/protocol.js). This is
// the rigorous "every scenario" gate for Stage 5 — it proves the messages the UI
// sends are accepted/rejected exactly as intended, over the real network
// transport, against the single ArchitectureStore.
//
// Requires a python (env PYTHON, default python3) with the backend deps
// installed, and a free :8765. If the backend can't run the whole suite skips
// (so it can't be falsely red on a machine without the deps).

import { describe, it, expect, beforeAll, afterAll, beforeEach } from "vitest";
import { spawn, execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { WebSocket } from "ws";

import * as proto from "../../src/protocol.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../../..");
// CI sets PYTHON=python3; local devs point it at their venv interpreter.
const PYTHON = process.env.PYTHON || "python3";
const URL = "ws://localhost:8765/ws";

function backendRunnable() {
  try {
    execFileSync(PYTHON, ["-c", "import fastapi, uvicorn, mcp"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

const haveBackend = existsSync(path.join(repoRoot, "backend", "main.py")) && backendRunnable();
const suite = haveBackend ? describe : describe.skip;

// --- a tiny promise-based ws client -------------------------------------------
class Client {
  constructor(ws) {
    this.ws = ws;
    this.queue = [];
    this.waiters = [];
    ws.on("message", (raw) => {
      const msg = JSON.parse(raw.toString());
      const w = this.waiters.shift();
      if (w) w(msg);
      else this.queue.push(msg);
    });
  }
  next(timeoutMs = 4000) {
    if (this.queue.length) return Promise.resolve(this.queue.shift());
    return new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error("timed out waiting for ws message")), timeoutMs);
      this.waiters.push((m) => {
        clearTimeout(t);
        resolve(m);
      });
    });
  }
  send(obj) {
    this.ws.send(JSON.stringify(obj));
  }
  sendRaw(text) {
    this.ws.send(text);
  }
  close() {
    this.ws.close();
  }
}

function open() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(URL);
    ws.once("open", () => resolve(new Client(ws)));
    ws.once("error", reject);
  });
}

async function waitForBackend(deadlineMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < deadlineMs) {
    try {
      const c = await open();
      c.close();
      return;
    } catch {
      await new Promise((r) => setTimeout(r, 250));
    }
  }
  throw new Error("backend did not come up on :8765");
}

// Mutation helpers asserting the §12 reply contract.
async function ok(c, message) {
  c.send(message);
  const ack = await c.next();
  expect(ack, `expected ack for ${message.type}`).toEqual({ type: "ack", ok: true });
  const state = await c.next();
  expect(state.type, `expected state broadcast after ${message.type}`).toBe("state");
  return state.data;
}
async function rejected(c, message) {
  c.send(message);
  const reply = await c.next();
  expect(reply.type, `expected error for ${message.type}`).toBe("error");
  expect(typeof reply.message).toBe("string");
  return reply.message;
}

// A clean connection: consume the on-connect state, reset the shared store, and
// return the client sitting on an empty architecture.
async function connectClean() {
  const c = await open();
  const initial = await c.next();
  expect(initial.type).toBe("state");
  await ok(c, proto.resetArchitecture());
  return c;
}

let proc;

suite("websocket protocol (live backend)", () => {
  beforeAll(async () => {
    proc = spawn(PYTHON, ["backend/main.py"], {
      cwd: repoRoot,
      env: { ...process.env, MODE: "standalone", LOG_LEVEL: "WARNING" },
      stdio: ["ignore", "ignore", "inherit"],
    });
    await waitForBackend();
  });

  afterAll(() => {
    if (proc && !proc.killed) proc.kill("SIGTERM");
  });

  let c;
  beforeEach(async () => {
    c = await connectClean();
  });

  // -- the happy path: every mutation, end to end ------------------------------
  it("initial state of a clean store is empty", async () => {
    const fresh = await open();
    const state = await fresh.next();
    fresh.close();
    expect(state.type).toBe("state");
    expect(state.data).toHaveProperty("nodes");
    expect(state.data).toHaveProperty("edges");
  });

  it("add_layer assigns incremental ids and broadcasts state", async () => {
    let s = await ok(c, proto.addLayer("input", { shape: [1, 28, 28], dtype: "float32" }));
    expect(s.nodes.map((n) => n.id)).toEqual(["n1"]);
    expect(s.nodes[0]).toMatchObject({ id: "n1", type: "input" });

    s = await ok(c, proto.addLayer("conv2d", { in_channels: 1, out_channels: 16, kernel_size: 3 }));
    expect(s.nodes.map((n) => n.id)).toEqual(["n1", "n2"]);
    // optional params get defaults applied by the catalog
    expect(s.nodes[1].params).toMatchObject({ stride: 1, padding: 0, dilation: 1 });
  });

  it("connect_layers adds an edge in insertion order", async () => {
    await ok(c, proto.addLayer("input", { shape: [1, 28, 28], dtype: "float32" }));
    await ok(c, proto.addLayer("conv2d", { in_channels: 1, out_channels: 16, kernel_size: 3 }));
    const s = await ok(c, proto.connectLayers("n1", "n2"));
    expect(s.edges).toEqual([{ from: "n1", to: "n2" }]);
  });

  it("update_layer merges params (partial update)", async () => {
    await ok(c, proto.addLayer("conv2d", { in_channels: 1, out_channels: 16, kernel_size: 3 }));
    const s = await ok(c, proto.updateLayer("n1", { out_channels: 32 }));
    expect(s.nodes[0].params).toMatchObject({ in_channels: 1, out_channels: 32, kernel_size: 3 });
  });

  it("disconnect_layers removes the edge", async () => {
    await ok(c, proto.addLayer("input", { shape: [1, 28, 28], dtype: "float32" }));
    await ok(c, proto.addLayer("relu", {}));
    await ok(c, proto.connectLayers("n1", "n2"));
    const s = await ok(c, proto.disconnectLayers("n1", "n2"));
    expect(s.edges).toEqual([]);
  });

  it("remove_layer cascades its edges", async () => {
    await ok(c, proto.addLayer("input", { shape: [1, 28, 28], dtype: "float32" }));
    await ok(c, proto.addLayer("relu", {}));
    await ok(c, proto.connectLayers("n1", "n2"));
    const s = await ok(c, proto.removeLayer("n2"));
    expect(s.nodes.map((n) => n.id)).toEqual(["n1"]);
    expect(s.edges).toEqual([]);
  });

  it("a merge node accepts two incoming edges", async () => {
    await ok(c, proto.addLayer("input", { shape: [1, 8], dtype: "float32" })); // n1
    await ok(c, proto.addLayer("relu", {})); // n2
    await ok(c, proto.addLayer("relu", {})); // n3
    await ok(c, proto.addLayer("add", {})); // n4
    await ok(c, proto.connectLayers("n1", "n2"));
    await ok(c, proto.connectLayers("n1", "n3"));
    await ok(c, proto.connectLayers("n2", "n4"));
    const s = await ok(c, proto.connectLayers("n3", "n4"));
    expect(s.edges.filter((e) => e.to === "n4")).toHaveLength(2);
  });

  it("reset_architecture clears the store; next id is n1 again", async () => {
    await ok(c, proto.addLayer("relu", {}));
    await ok(c, proto.addLayer("relu", {}));
    await ok(c, proto.resetArchitecture());
    const s = await ok(c, proto.addLayer("relu", {}));
    expect(s.nodes.map((n) => n.id)).toEqual(["n1"]);
  });

  // -- every rejection branch (error reply, socket stays alive) ----------------
  it("rejects an unknown layer type", async () => {
    expect(await rejected(c, proto.addLayer("bogus", {}))).toMatch(/unknown layer type/);
  });

  it("rejects add_layer missing a required param", async () => {
    expect(await rejected(c, proto.addLayer("conv2d", {}))).toMatch(/missing required param/);
  });

  it("rejects add_layer with an unknown param", async () => {
    expect(await rejected(c, proto.addLayer("relu", { nope: 1 }))).toMatch(/unknown param/);
  });

  it("rejects a second input node", async () => {
    await ok(c, proto.addLayer("input", { shape: [1, 4], dtype: "float32" }));
    expect(await rejected(c, proto.addLayer("input", { shape: [1, 4], dtype: "float32" }))).toMatch(
      /input node already exists/,
    );
  });

  it("rejects a self-loop", async () => {
    await ok(c, proto.addLayer("relu", {}));
    expect(await rejected(c, proto.connectLayers("n1", "n1"))).toMatch(/self-loop/);
  });

  it("rejects connecting to an unknown target and from an unknown source", async () => {
    await ok(c, proto.addLayer("relu", {}));
    expect(await rejected(c, proto.connectLayers("n1", "n999"))).toMatch(/unknown target/);
    expect(await rejected(c, proto.connectLayers("n999", "n1"))).toMatch(/unknown source/);
  });

  it("rejects an edge into the input node", async () => {
    await ok(c, proto.addLayer("input", { shape: [1, 4], dtype: "float32" })); // n1
    await ok(c, proto.addLayer("relu", {})); // n2
    expect(await rejected(c, proto.connectLayers("n2", "n1"))).toMatch(/input node/);
  });

  it("rejects a duplicate edge", async () => {
    await ok(c, proto.addLayer("input", { shape: [1, 4], dtype: "float32" }));
    await ok(c, proto.addLayer("relu", {}));
    await ok(c, proto.connectLayers("n1", "n2"));
    expect(await rejected(c, proto.connectLayers("n1", "n2"))).toMatch(/already exists/);
  });

  it("rejects a second input into a non-merge node", async () => {
    await ok(c, proto.addLayer("input", { shape: [1, 4], dtype: "float32" })); // n1
    await ok(c, proto.addLayer("relu", {})); // n2
    await ok(c, proto.addLayer("relu", {})); // n3 -> target (non-merge)
    await ok(c, proto.connectLayers("n1", "n3"));
    expect(await rejected(c, proto.connectLayers("n2", "n3"))).toMatch(/already has an incoming edge/);
  });

  it("rejects an edge that would create a cycle", async () => {
    await ok(c, proto.addLayer("relu", {})); // n1
    await ok(c, proto.addLayer("relu", {})); // n2
    await ok(c, proto.addLayer("relu", {})); // n3
    await ok(c, proto.connectLayers("n1", "n2"));
    await ok(c, proto.connectLayers("n2", "n3"));
    expect(await rejected(c, proto.connectLayers("n3", "n1"))).toMatch(/cycle/);
  });

  it("rejects disconnecting a nonexistent edge", async () => {
    await ok(c, proto.addLayer("relu", {}));
    await ok(c, proto.addLayer("relu", {}));
    expect(await rejected(c, proto.disconnectLayers("n1", "n2"))).toMatch(/does not exist/);
  });

  it("rejects updating/removing an unknown node", async () => {
    expect(await rejected(c, proto.updateLayer("n999", { p: 0.1 }))).toMatch(/unknown node/);
    expect(await rejected(c, proto.removeLayer("n999"))).toMatch(/unknown node/);
  });

  // -- malformed inputs never drop the socket ----------------------------------
  it("rejects malformed json, non-objects, unknown types, and bad params without dropping the socket", async () => {
    c.sendRaw("not json{");
    expect((await c.next()).type).toBe("error");

    c.sendRaw(JSON.stringify(["not", "an", "object"]));
    expect((await c.next()).type).toBe("error");

    c.send({ type: "frobnicate" });
    expect((await c.next()).type).toBe("error");

    c.send({ type: "add_layer", layer_type: "relu" }); // missing params field
    expect((await c.next()).type).toBe("error");

    c.send({ type: "add_layer", layer_type: "relu", params: null }); // non-dict params
    expect((await c.next()).type).toBe("error");

    // socket is still alive: a valid mutation still works
    const s = await ok(c, proto.addLayer("relu", {}));
    expect(s.nodes).toHaveLength(1);
  });

  // -- multi-client broadcast fan-out ------------------------------------------
  it("broadcasts state to every connected client after a mutation", async () => {
    const a = await connectClean();
    const b = await open();
    expect((await b.next()).type).toBe("state"); // b's on-connect state

    a.send(proto.addLayer("relu", {}));
    expect((await a.next()).type).toBe("ack");
    const aState = await a.next();
    const bState = await b.next();
    expect(aState.type).toBe("state");
    expect(bState.type).toBe("state");
    expect(bState.data.nodes).toHaveLength(1);
    a.close();
    b.close();
  });
});
