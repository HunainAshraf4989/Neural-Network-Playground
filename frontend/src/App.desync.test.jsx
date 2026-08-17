import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

// Regression guard for the canvas "disappearing edges/nodes" class of bug.
//
// Unlike App.test.jsx's inert stand-in, this React Flow mock *applies* change
// events to controlled state the way the real one does (a `remove` change drops
// the element). It also exposes buttons that fire a spurious "remove everything"
// change stream - exactly what React Flow does internally when it can't resolve
// an element for a frame. App must drop those (backend owns existence), so the
// canvas keeps showing what the backend still has.
function applyMini(changes, items) {
  const removed = new Set(changes.filter((c) => c.type === "remove").map((c) => c.id));
  return items.filter((it) => !removed.has(it.id));
}

vi.mock("@xyflow/react", async () => {
  const React = await import("react");
  const { useState } = React;
  const stateHook = (init) => {
    const [s, set] = useState(init);
    const onChange = (changes) => set((cur) => applyMini(changes, cur));
    return [s, set, onChange];
  };
  return {
    ReactFlow: ({ nodes = [], edges = [], onNodesChange, onEdgesChange, children }) =>
      React.createElement(
        "div",
        { "data-testid": "rf" },
        React.createElement("button", {
          "data-testid": "rf-remove-all-edges",
          onClick: () => onEdgesChange?.(edges.map((e) => ({ type: "remove", id: e.id }))),
        }),
        React.createElement("button", {
          "data-testid": "rf-remove-all-nodes",
          onClick: () => onNodesChange?.(nodes.map((n) => ({ type: "remove", id: n.id }))),
        }),
        edges.map((e) => React.createElement("div", { key: e.id, "data-testid": `edge-${e.id}` }, e.id)),
        nodes.map((n) => React.createElement("div", { key: n.id, "data-testid": `node-${n.id}` }, n.id)),
        children,
      ),
    Background: () => null,
    Controls: () => null,
    useNodesState: stateHook,
    useEdgesState: stateHook,
    getNodesBounds: () => ({ x: 0, y: 0, width: 0, height: 0 }),
    getViewportForBounds: () => ({ x: 0, y: 0, zoom: 1 }),
  };
});

vi.mock("./useArchitectureSocket.js", () => ({ useArchitectureSocket: vi.fn() }));

import { useArchitectureSocket } from "./useArchitectureSocket.js";
import App from "./App.jsx";

const ARCH = {
  nodes: [
    { id: "n1", type: "input", params: {} },
    { id: "n2", type: "conv2d", params: {} },
  ],
  edges: [{ from: "n1", to: "n2" }],
  layout: {},
};

function mockHook(arch) {
  useArchitectureSocket.mockReturnValue({
    arch,
    connected: true,
    notice: null,
    send: vi.fn(),
    requestCode: vi.fn(),
    clearNotice: () => {},
  });
}

describe("canvas existence is backend-owned", () => {
  beforeEach(() => vi.clearAllMocks());

  it("keeps edges when React Flow emits a spurious remove change", () => {
    mockHook(ARCH);
    render(<App />);
    expect(screen.getByTestId("edge-n1->n2")).toBeInTheDocument();
    // Spurious internal remove (no backend round-trip) must be ignored.
    fireEvent.click(screen.getByTestId("rf-remove-all-edges"));
    expect(screen.getByTestId("edge-n1->n2")).toBeInTheDocument();
  });

  it("keeps nodes when React Flow emits a spurious remove change", () => {
    mockHook(ARCH);
    render(<App />);
    expect(screen.getByTestId("node-n2")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("rf-remove-all-nodes"));
    expect(screen.getByTestId("node-n2")).toBeInTheDocument();
  });
});
