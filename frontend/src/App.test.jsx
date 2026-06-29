import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

// Mock React Flow with a tiny stand-in: render one button per node that fires
// onNodeClick, so we can exercise selection without a real canvas.
vi.mock("@xyflow/react", async () => {
  const React = await import("react");
  const { useState } = React;
  const mk = (init) => {
    const [s, set] = useState(init);
    return [s, set, () => {}];
  };
  return {
    ReactFlow: ({ nodes = [], onNodeClick, children }) =>
      React.createElement(
        "div",
        { "data-testid": "rf" },
        nodes.map((n) =>
          React.createElement(
            "button",
            { key: n.id, "data-testid": `node-${n.id}`, onClick: (e) => onNodeClick?.(e, n) },
            n.data.type,
          ),
        ),
        children,
      ),
    Background: () => null,
    Controls: () => null,
    useNodesState: mk,
    useEdgesState: mk,
    getNodesBounds: () => ({ x: 0, y: 0, width: 0, height: 0 }),
    getViewportForBounds: () => ({ x: 0, y: 0, zoom: 1 }),
  };
});

vi.mock("./useArchitectureSocket.js", () => ({ useArchitectureSocket: vi.fn() }));

import { useArchitectureSocket } from "./useArchitectureSocket.js";
import App from "./App.jsx";

function mockHook(overrides = {}) {
  const send = overrides.send ?? vi.fn();
  useArchitectureSocket.mockReturnValue({
    arch: { nodes: [], edges: [] },
    connected: true,
    notice: null,
    clearNotice: () => {},
    ...overrides,
    send, // forced last so a `send` in overrides can't shadow the captured mock
  });
  return send;
}

describe("App", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows connection status", () => {
    mockHook({ connected: true });
    const { rerender } = render(<App />);
    expect(screen.getByText("connected")).toBeInTheDocument();

    mockHook({ connected: false });
    rerender(<App />);
    expect(screen.getByText("disconnected")).toBeInTheDocument();
  });

  it("renders a node from the broadcast architecture", () => {
    mockHook({ arch: { nodes: [{ id: "n1", type: "conv2d", params: {} }], edges: [] } });
    render(<App />);
    expect(screen.getByTestId("node-n1")).toHaveTextContent("conv2d");
  });

  it("sends add_layer when a palette item is clicked", () => {
    const send = mockHook();
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Add relu" }));
    expect(send).toHaveBeenCalledWith({ type: "add_layer", layer_type: "relu", params: {} });
  });

  it("sends reset_architecture when Reset is clicked", () => {
    const send = mockHook();
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(send).toHaveBeenCalledWith({ type: "reset_architecture" });
  });

  it("selecting a node opens the params panel and Save sends update_layer", () => {
    const send = mockHook({
      arch: { nodes: [{ id: "n1", type: "dropout", params: { p: 0.5 } }], edges: [] },
    });
    render(<App />);
    fireEvent.click(screen.getByTestId("node-n1"));
    // panel now shows; Save sends update_layer with the (unchanged) params
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(send).toHaveBeenCalledWith({ type: "update_layer", node_id: "n1", params: { p: 0.5 } });
  });

  it("Delete in the params panel sends remove_layer and closes the panel", () => {
    const send = mockHook({
      arch: { nodes: [{ id: "n1", type: "relu", params: {} }], edges: [] },
    });
    render(<App />);
    fireEvent.click(screen.getByTestId("node-n1"));
    expect(screen.getByText("relu", { selector: ".params__title" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(send).toHaveBeenCalledWith({ type: "remove_layer", node_id: "n1" });
    expect(screen.queryByText("relu", { selector: ".params__title" })).not.toBeInTheDocument();
  });

  it("surfaces an error notice from the backend", () => {
    mockHook({ notice: { kind: "error", message: "cycle rejected" } });
    render(<App />);
    expect(screen.getByText("cycle rejected")).toBeInTheDocument();
  });
});
