import { describe, it, expect } from "vitest";
import * as proto from "./protocol.js";

// These shapes must match backend ws_app._DISPATCH exactly, or the backend
// rejects them. The e2e suite proves it against the live server; this pins the
// JSON shape cheaply.
describe("protocol builders", () => {
  it("add_layer", () => {
    expect(proto.addLayer("conv2d", { in_channels: 1 })).toEqual({
      type: "add_layer",
      layer_type: "conv2d",
      params: { in_channels: 1 },
    });
  });

  it("update_layer", () => {
    expect(proto.updateLayer("n3", { p: 0.2 })).toEqual({
      type: "update_layer",
      node_id: "n3",
      params: { p: 0.2 },
    });
  });

  it("remove_layer", () => {
    expect(proto.removeLayer("n5")).toEqual({ type: "remove_layer", node_id: "n5" });
  });

  it("connect_layers", () => {
    expect(proto.connectLayers("n1", "n2")).toEqual({
      type: "connect_layers",
      from_id: "n1",
      to_id: "n2",
    });
  });

  it("disconnect_layers", () => {
    expect(proto.disconnectLayers("n1", "n2")).toEqual({
      type: "disconnect_layers",
      from_id: "n1",
      to_id: "n2",
    });
  });

  it("ungroup", () => {
    expect(proto.ungroup("n4")).toEqual({ type: "ungroup", node_id: "n4" });
  });

  it("reset_architecture", () => {
    expect(proto.resetArchitecture()).toEqual({ type: "reset_architecture" });
  });

  it("generate_code", () => {
    expect(proto.generateCode()).toEqual({ type: "generate_code" });
  });
});
