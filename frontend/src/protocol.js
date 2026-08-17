// Client → server message builders (spec §12). One builder per mutating store
// method; each returns exactly the JSON shape the ws handler's _DISPATCH expects.
// Keeping these pure makes them unit-testable and lets the e2e suite drive the
// real backend through the very same code path the UI uses.

export function addLayer(layer_type, params) {
  return { type: "add_layer", layer_type, params };
}

export function updateLayer(node_id, params) {
  return { type: "update_layer", node_id, params };
}

export function removeLayer(node_id) {
  return { type: "remove_layer", node_id };
}

export function connectLayers(from_id, to_id) {
  return { type: "connect_layers", from_id, to_id };
}

export function disconnectLayers(from_id, to_id) {
  return { type: "disconnect_layers", from_id, to_id };
}

// Ungroup: ask the backend to replace a composite node (transformer block,
// stacked/bidirectional RNN) with its real editable sub-layers - a persistent
// store mutation, unlike the read-only Expand view.
export function ungroup(node_id) {
  return { type: "ungroup", node_id };
}

export function resetArchitecture() {
  return { type: "reset_architecture" };
}

// Read-only request: asks the backend to emit standalone PyTorch source for the
// current graph. The reply is a `{type:"code"}` message, not a state broadcast.
export function generateCode() {
  return { type: "generate_code" };
}
