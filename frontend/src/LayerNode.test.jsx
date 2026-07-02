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
    // a quiet op renders at the small fixed diameter (64) inside the 150 cell, so
    // each handle is pushed in to (150 - 64) / 2 = 43px rather than floating at the
    // cell edge away from the visible shape.
    render(<LayerNode data={{ type: "relu", category: "activation", params: {} }} />);
    expect(screen.getByTestId("handle-target").dataset.inset).toBe("43");
    expect(screen.getByTestId("handle-source").dataset.inset).toBe("43");
  });

  it("renders null params as 'auto' and arrays bracketed", () => {
    render(
      <LayerNode data={{ type: "maxpool2d", category: "pooling", params: { kernel_size: 2, stride: null } }} />,
    );
    expect(screen.getByText("auto")).toBeInTheDocument();
  });

  it("collapsed, a linear is a compact pill (no neuron column)", () => {
    const { container } = render(
      <LayerNode data={{ type: "linear", category: "linear", params: { out_features: 768 }, neurons: 6, width: 768 }} />,
    );
    expect(container.querySelector(".nn-glyph--stack")).toBeInTheDocument();
    expect(container.querySelector(".nn-neuron-col")).not.toBeInTheDocument();
  });

  it("expanded, a linear becomes a column of `neurons` solid circles", () => {
    const { container } = render(
      <LayerNode data={{ type: "linear", category: "linear", params: { out_features: 768 }, neurons: 6, width: 768, expanded: true }} />,
    );
    expect(container.querySelector(".nn-node--neurons")).toBeInTheDocument();
    expect(container.querySelectorAll(".nn-neuron-col i")).toHaveLength(6);
    expect(container.querySelector(".nn-glyph--stack")).not.toBeInTheDocument();
  });

  it("shows a ⋮ when the layer's true width exceeds the circles drawn", () => {
    const { container } = render(
      <LayerNode data={{ type: "linear", category: "linear", params: { out_features: 768 }, neurons: 6, width: 768, expanded: true }} />,
    );
    // 6 circles remain, with the ⋮ (three small dots) inserted for the omitted middle.
    expect(container.querySelector(".nn-neuron-col__more")).toBeInTheDocument();
    expect(container.querySelectorAll(".nn-neuron-col i")).toHaveLength(6);
  });

  it("draws no ⋮ when the layer's width equals the circles (nothing omitted)", () => {
    const { container } = render(
      <LayerNode data={{ type: "linear", category: "linear", params: { out_features: 3 }, neurons: 3, width: 3, expanded: true }} />,
    );
    expect(container.querySelector(".nn-neuron-col__more")).not.toBeInTheDocument();
    expect(container.querySelectorAll(".nn-neuron-col i")).toHaveLength(3);
  });

  it("expanded, an activation (relu) is also drawn as a neuron column", () => {
    const { container } = render(
      <LayerNode data={{ type: "relu", category: "activation", params: {}, neurons: 5, width: 512, expanded: true }} />,
    );
    expect(container.querySelector(".nn-node--neurons")).toBeInTheDocument();
    expect(container.querySelectorAll(".nn-neuron-col i")).toHaveLength(5);
  });

  it("a synthetic linear inside an expanded composite stays a pill, not neurons", () => {
    const { container } = render(
      <LayerNode data={{ type: "linear", category: "linear", params: {}, neurons: 5, width: 128, expanded: true, synthetic: true }} />,
    );
    expect(container.querySelector(".nn-neuron-col")).not.toBeInTheDocument();
    expect(container.querySelector(".nn-glyph--stack")).toBeInTheDocument();
  });
});
