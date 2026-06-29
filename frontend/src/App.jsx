// Top-level app: renders the shared architecture on a React Flow canvas and
// turns user actions into §12 client messages. The backend is the single source
// of truth — every mutation is sent, never applied locally; the canvas only ever
// shows what the server broadcasts back (so it can't drift from the agent's view).

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import LayerNode from "./LayerNode.jsx";
import LayerPalette from "./LayerPalette.jsx";
import ParamsPanel from "./ParamsPanel.jsx";
import { useArchitectureSocket } from "./useArchitectureSocket.js";
import { toFlowElements } from "./flow.js";
import { defaultParams } from "./catalog.js";
import * as proto from "./protocol.js";

const NODE_TYPES = { layer: LayerNode };

export default function App() {
  const { arch, connected, notice, send } = useArchitectureSocket();
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedId, setSelectedId] = useState(null);
  const wrapperRef = useRef(null);

  // Server state is authoritative: recompute the whole canvas from each `state`
  // broadcast (auto-layout, no persisted x/y per spec §13).
  useEffect(() => {
    const { nodes: n, edges: e } = toFlowElements(arch);
    setNodes(n);
    setEdges(e);
  }, [arch, setNodes, setEdges]);

  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedId) ?? null,
    [nodes, selectedId],
  );

  const add = useCallback((type) => send(proto.addLayer(type, defaultParams(type))), [send]);

  const onConnect = useCallback(
    (c) => send(proto.connectLayers(c.source, c.target)),
    [send],
  );
  const onNodesDelete = useCallback(
    (deleted) => deleted.forEach((n) => send(proto.removeLayer(n.id))),
    [send],
  );
  const onEdgesDelete = useCallback(
    (deleted) => deleted.forEach((e) => send(proto.disconnectLayers(e.source, e.target))),
    [send],
  );

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();
      const type = event.dataTransfer.getData("application/nn-layer-type");
      if (type) add(type);
    },
    [add],
  );
  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  }, []);

  return (
    <div className="app">
      <LayerPalette onAdd={add} />

      <div className="canvas" ref={wrapperRef} onDrop={onDrop} onDragOver={onDragOver}>
        <div className="toolbar">
          <span className={`status status--${connected ? "ok" : "down"}`}>
            {connected ? "connected" : "disconnected"}
          </span>
          <button type="button" className="toolbar__reset" onClick={() => send(proto.resetArchitecture())}>
            Reset
          </button>
          {notice && <span className={`notice notice--${notice.kind}`}>{notice.message}</span>}
        </div>

        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={NODE_TYPES}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodesDelete={onNodesDelete}
          onEdgesDelete={onEdgesDelete}
          onNodeClick={(_, node) => setSelectedId(node.id)}
          onPaneClick={() => setSelectedId(null)}
          nodesDraggable={false}
          fitView
        >
          <Background />
          <Controls />
        </ReactFlow>
      </div>

      <ParamsPanel
        node={selectedNode}
        onSave={(id, params) => send(proto.updateLayer(id, params))}
        onDelete={(id) => {
          send(proto.removeLayer(id));
          setSelectedId(null);
        }}
        onClose={() => setSelectedId(null)}
      />
    </div>
  );
}
