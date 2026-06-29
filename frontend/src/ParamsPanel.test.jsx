import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ParamsPanel from "./ParamsPanel.jsx";

const node = (type, params) => ({ id: "n2", data: { type, params } });

describe("ParamsPanel", () => {
  it("renders nothing when no node is selected", () => {
    const { container } = render(<ParamsPanel node={null} onSave={() => {}} onDelete={() => {}} onClose={() => {}} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the type, id and one field per param", () => {
    render(<ParamsPanel node={node("dropout", { p: 0.5 })} onSave={() => {}} onDelete={() => {}} onClose={() => {}} />);
    expect(screen.getByText("dropout")).toBeInTheDocument();
    expect(screen.getByText("n2")).toBeInTheDocument();
    expect(screen.getByText("p")).toBeInTheDocument();
  });

  it("shows 'No parameters.' for a param-less layer", () => {
    render(<ParamsPanel node={node("relu", {})} onSave={() => {}} onDelete={() => {}} onClose={() => {}} />);
    expect(screen.getByText("No parameters.")).toBeInTheDocument();
  });

  it("saves coerced numeric params via update_layer", () => {
    const onSave = vi.fn();
    render(<ParamsPanel node={node("dropout", { p: 0.5 })} onSave={onSave} onDelete={() => {}} onClose={() => {}} />);
    const input = screen.getByDisplayValue("0.5");
    fireEvent.change(input, { target: { value: "0.2" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onSave).toHaveBeenCalledWith("n2", { p: 0.2 });
  });

  it("saves a list param as a numeric array", () => {
    const onSave = vi.fn();
    render(
      <ParamsPanel
        node={node("input", { shape: [1, 28, 28], dtype: "float32" })}
        onSave={onSave}
        onDelete={() => {}}
        onClose={() => {}}
      />,
    );
    fireEvent.change(screen.getByDisplayValue("1,28,28"), { target: { value: "3,32,32" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onSave).toHaveBeenCalledWith("n2", { shape: [3, 32, 32], dtype: "float32" });
  });

  it("renders a boolean param as a checkbox and saves the toggle", () => {
    const onSave = vi.fn();
    render(
      <ParamsPanel
        node={node("lstm", { input_size: 64, hidden_size: 128, num_layers: 1, bidirectional: false })}
        onSave={onSave}
        onDelete={() => {}}
        onClose={() => {}}
      />,
    );
    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).not.toBeChecked();
    fireEvent.click(checkbox);
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onSave).toHaveBeenCalledWith("n2", {
      input_size: 64,
      hidden_size: 128,
      num_layers: 1,
      bidirectional: true,
    });
  });

  it("calls onDelete with the node id", () => {
    const onDelete = vi.fn();
    render(<ParamsPanel node={node("relu", {})} onSave={() => {}} onDelete={onDelete} onClose={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(onDelete).toHaveBeenCalledWith("n2");
  });

  it("re-seeds the form when a different node is selected", () => {
    const { rerender } = render(
      <ParamsPanel node={node("dropout", { p: 0.5 })} onSave={() => {}} onDelete={() => {}} onClose={() => {}} />,
    );
    expect(screen.getByDisplayValue("0.5")).toBeInTheDocument();
    rerender(
      <ParamsPanel
        node={{ id: "n3", data: { type: "dropout", params: { p: 0.9 } } }}
        onSave={() => {}}
        onDelete={() => {}}
        onClose={() => {}}
      />,
    );
    expect(screen.getByDisplayValue("0.9")).toBeInTheDocument();
  });
});
