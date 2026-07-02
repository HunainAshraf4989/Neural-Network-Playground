import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import NeuronBundleEdge from "./NeuronBundleEdge.jsx";

describe("NeuronBundleEdge", () => {
  it("draws sourceCount × targetCount fully-connected segments", () => {
    const { container } = render(
      <svg>
        <NeuronBundleEdge sourceX={0} sourceY={100} targetX={200} targetY={100} data={{ sourceCount: 3, targetCount: 4 }} />
      </svg>,
    );
    const path = container.querySelector("path.nn-bundle");
    expect(path).toBeInTheDocument();
    const d = path.getAttribute("d");
    expect((d.match(/M/g) || []).length).toBe(12); // 3 × 4 lines
    expect((d.match(/L/g) || []).length).toBe(12);
  });

  it("still draws count×count segments when a column is truncated (⋮ slot omitted)", () => {
    const { container } = render(
      <svg>
        <NeuronBundleEdge
          sourceX={0} sourceY={100} targetX={200} targetY={100}
          data={{ sourceCount: 6, targetCount: 6, sourceTruncated: true, targetTruncated: true }}
        />
      </svg>,
    );
    const d = container.querySelector("path.nn-bundle").getAttribute("d");
    // 6 endpoints per side despite the ⋮; endpoints land off-center (no line at y=100).
    expect((d.match(/M/g) || []).length).toBe(36);
    expect(d).not.toContain("M0,100");
  });

  it("falls back to a single line when counts are missing", () => {
    const { container } = render(
      <svg>
        <NeuronBundleEdge sourceX={0} sourceY={0} targetX={10} targetY={0} data={{}} />
      </svg>,
    );
    const d = container.querySelector("path.nn-bundle").getAttribute("d");
    expect((d.match(/M/g) || []).length).toBe(1);
  });
});
