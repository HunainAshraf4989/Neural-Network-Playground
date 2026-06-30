import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import LayerNode from "./LayerNode.jsx";

// Render React Flow handles as simple markers so we can assert on them without a
// full React Flow provider.
vi.mock("@xyflow/react", () => ({
  Position: { Left: "left", Right: "right" },
  Handle: ({ type, position, style }) => (
    <div data-testid={`handle-${type}`} data-position={position} data-inset={style?.left ?? style?.right} />
  ),
}));

describe("LayerNode", () => {
  it("renders the type name and a few params", () => {
    render(
      <LayerNode
        data={{ type: "conv2d", category: "conv", params: { in_channels: 1, out_channels: 16, kernel_size: 3 } }}
      />,
    );
    expect(screen.getByText("conv2d")).toBeInTheDocument();
    expect(screen.getByText("in_channels")).toBeInTheDocument();
    expect(screen.getByText("16")).toBeInTheDocument();
  });

  it("gives an input node a source handle but NO target handle", () => {
    render(<LayerNode data={{ type: "input", category: "input", params: { dtype: "float32" } }} />);
    expect(screen.queryByTestId("handle-target")).not.toBeInTheDocument();
    expect(screen.getByTestId("handle-source")).toBeInTheDocument();
  });

  it("gives every non-input node both a target and a source handle", () => {
    render(<LayerNode data={{ type: "relu", category: "activation", params: {} }} />);
    expect(screen.getByTestId("handle-target")).toBeInTheDocument();
    expect(screen.getByTestId("handle-source")).toBeInTheDocument();
  });

  it("insets connection handles to the glyph edge so small nodes stay easy to wire", () => {
    // a quiet op renders at the small fixed diameter (46) inside the 116 cell, so
    // each handle is pushed in to (116 - 46) / 2 = 35px rather than floating at the
    // cell edge away from the visible shape.
    render(<LayerNode data={{ type: "relu", category: "activation", params: {} }} />);
    expect(screen.getByTestId("handle-target").dataset.inset).toBe("35");
    expect(screen.getByTestId("handle-source").dataset.inset).toBe("35");
  });

  it("renders null params as 'auto' and arrays bracketed", () => {
    render(
      <LayerNode data={{ type: "maxpool2d", category: "pooling", params: { kernel_size: 2, stride: null } }} />,
    );
    expect(screen.getByText("auto")).toBeInTheDocument();
  });
});
