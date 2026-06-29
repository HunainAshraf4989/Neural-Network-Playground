import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import LayerPalette from "./LayerPalette.jsx";
import { CATALOG } from "./catalog.js";

describe("LayerPalette", () => {
  it("renders a button for every catalog type", () => {
    render(<LayerPalette onAdd={() => {}} />);
    for (const type of Object.keys(CATALOG)) {
      expect(screen.getByRole("button", { name: `Add ${type}` })).toBeInTheDocument();
    }
  });

  it("renders the category group headings", () => {
    render(<LayerPalette onAdd={() => {}} />);
    // a representative sample of categories
    for (const c of ["conv", "pooling", "activation", "merge"]) {
      expect(screen.getByText(c)).toBeInTheDocument();
    }
  });

  it("calls onAdd with the type when an item is clicked", () => {
    const onAdd = vi.fn();
    render(<LayerPalette onAdd={onAdd} />);
    fireEvent.click(screen.getByRole("button", { name: "Add conv2d" }));
    expect(onAdd).toHaveBeenCalledWith("conv2d");
  });

  it("puts the layer type on the dataTransfer when dragging starts", () => {
    render(<LayerPalette onAdd={() => {}} />);
    const setData = vi.fn();
    fireEvent.dragStart(screen.getByRole("button", { name: "Add lstm" }), {
      dataTransfer: { setData, effectAllowed: "" },
    });
    expect(setData).toHaveBeenCalledWith("application/nn-layer-type", "lstm");
  });
});
