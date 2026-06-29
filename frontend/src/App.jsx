// Top-level app: renders the shared architecture on a React Flow canvas and
// turns user actions into §12 client messages. The backend is the single source
// of truth — every mutation is sent, never applied locally; the canvas only ever
// shows what the server broadcasts back (so it can't drift from the agent's view).
//
// Layout note (CLAUDE.md invariant 8): node x/y are NOT in the shared schema —
// they're recomputed each broadcast by the deterministic auto-layout. But users
// can drag nodes, so we keep a *frontend-only* override map of positions the user
// set (drag or drop-at-cursor) and apply it on top of the auto-layout. New nodes
// still auto-layout; nothing about positions ever reaches the backend.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  getNodesBounds,
  getViewportForBounds,
} from "@xyflow/react";
import { toPng } from "html-to-image";
import "@xyflow/react/dist/style.css";

import LayerNode from "./LayerNode.jsx";
import LayerPalette from "./LayerPalette.jsx";
import ParamsPanel from "./ParamsPanel.jsx";
import DeletableEdge from "./DeletableEdge.jsx";
import CodeModal from "./CodeModal.jsx";
import { useArchitectureSocket } from "./useArchitectureSocket.js";
import { toFlowElements } from "./flow.js";
import { COL_GAP, ROW_GAP, computeLayout, colOf, rowOf, cellPos, snapToCell } from "./layout.js";
import { defaultParams } from "./catalog.js";
import * as proto from "./protocol.js";

const NODE_TYPES = { layer: LayerNode };
const EDGE_TYPES = { deletable: DeletableEdge };
const IMAGE_BG = "#0f1420"; // matches the dark canvas background
// Dragging snaps onto the column/row grid so nodes stay aligned into clean columns.
const GRID = [COL_GAP, ROW_GAP];

export default function App() {
  const { arch, connected, notice, send, requestCode } = useArchitectureSocket();
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [codeModal, setCodeModal] = useState(null); // {code} | {error} | null
  const rfRef = useRef(null);
  const prevNodeCount = useRef(0);
  const dragOverridesRef = useRef(new Map()); // id -> {x,y}: where the user dragged a node

  // The backend is authoritative for *which* nodes/edges exist; the frontend
  // owns pixel layout. Positions are recomputed every broadcast from the graph
  // itself (`computeLayout`: column = distance from input, so wiring n1->n2 puts
  // n2 one column right of n1), then two overrides are layered on top:
  //   1. a node the user dragged keeps its dragged position;
  //   2. else an agent placement hint ({col,row} on the backend `layout`) wins.
  // Because every node is positioned deterministically from the live graph each
  // time, nodes/edges can never desync or vanish from the canvas.
  useEffect(() => {
    const ids = new Set((arch?.nodes ?? []).map((n) => n.id));
    const { nodes: n, edges: e } = toFlowElements(arch);

    // Forget drag positions for nodes that no longer exist.
    for (const id of [...dragOverridesRef.current.keys()]) {
      if (!ids.has(id)) dragOverridesRef.current.delete(id);
    }

    const base = computeLayout(arch); // edge-driven columns for every node
    const hints = arch?.layout ?? {}; // agent placement hints (id -> {col,row})
    const positionFor = (id) => {
      if (dragOverridesRef.current.has(id)) return dragOverridesRef.current.get(id);
      const hint = hints[id];
      const b = base[id] ?? { x: 0, y: 0 };
      if (hint && (hint.col != null || hint.row != null)) {
        return cellPos(hint.col != null ? hint.col : colOf(b), hint.row != null ? hint.row : rowOf(b));
      }
      return b;
    };

    setNodes(n.map((node) => ({ ...node, position: positionFor(node.id) })));
    setEdges(
      e.map((edge) => ({
        ...edge,
        type: "deletable",
        data: { onDelete: () => send(proto.disconnectLayers(edge.source, edge.target)) },
      })),
    );
  }, [arch, setNodes, setEdges, send]);

  // Re-frame only when a node is ADDED, so a new node is always visible — but
  // leave the view alone on delete and on param edits (which previously caused
  // the canvas to jump/zoom unexpectedly). `maxZoom` keeps the fit from zooming
  // way in on a one/two-node graph (which made the circles huge).
  useEffect(() => {
    const count = arch?.nodes?.length ?? 0;
    const grew = count > prevNodeCount.current;
    prevNodeCount.current = count;
    if (!grew) return;
    if (rfRef.current) {
      requestAnimationFrame(() =>
        rfRef.current?.fitView({ padding: 0.25, maxZoom: 0.85, duration: 300 }),
      );
    }
  }, [arch]);

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

  // Remember where the user dragged a node (snapped to the grid) so it overrides
  // the edge-driven layout and stays put across broadcasts.
  const onNodeDragStop = useCallback((_, node) => {
    dragOverridesRef.current.set(node.id, snapToCell(node.position));
  }, []);

  // Dropping a palette item just adds the layer; the edge-driven layout places
  // it (in column 0 until it's wired, then it flows to the right of its source).
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

  const onExportCode = useCallback(async () => {
    setCodeModal(await requestCode());
  }, [requestCode]);

  // Export the canvas to PNG: frame the nodes' bounding box and rasterize the
  // React Flow viewport (the standard @xyflow recipe).
  const onDownloadImage = useCallback(() => {
    if (nodes.length === 0) return;
    const bounds = getNodesBounds(nodes);
    const pad = 80;
    const width = Math.max(bounds.width + pad * 2, 400);
    const height = Math.max(bounds.height + pad * 2, 300);
    const viewport = getViewportForBounds(bounds, width, height, 0.5, 2, 0.1);
    const el = document.querySelector(".react-flow__viewport");
    if (!el) return;
    toPng(el, {
      backgroundColor: IMAGE_BG,
      width,
      height,
      style: {
        width: `${width}px`,
        height: `${height}px`,
        transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.zoom})`,
      },
    }).then((dataUrl) => {
      const a = document.createElement("a");
      a.setAttribute("download", "neural-network.png");
      a.setAttribute("href", dataUrl);
      a.click();
    });
  }, [nodes]);

  return (
    <div className={`app${selectedNode ? " app--panel" : ""}`}>
      <LayerPalette onAdd={add} />

      <div className="canvas" onDrop={onDrop} onDragOver={onDragOver}>
        <div className="toolbar">
          <span className={`status status--${connected ? "ok" : "down"}`}>
            {connected ? "connected" : "disconnected"}
          </span>
          <button type="button" className="toolbar__btn" onClick={onExportCode} disabled={!connected}>
            Export code
          </button>
          <button type="button" className="toolbar__btn" onClick={onDownloadImage} disabled={nodes.length === 0}>
            Download PNG
          </button>
          <button type="button" className="toolbar__btn" onClick={() => send(proto.resetArchitecture())}>
            Reset
          </button>
          {notice && <span className={`notice notice--${notice.kind}`}>{notice.message}</span>}
        </div>

        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={NODE_TYPES}
          edgeTypes={EDGE_TYPES}
          onInit={(inst) => (rfRef.current = inst)}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodesDelete={onNodesDelete}
          onEdgesDelete={onEdgesDelete}
          onNodeDragStop={onNodeDragStop}
          onNodeClick={(_, node) => setSelectedId(node.id)}
          onPaneClick={() => setSelectedId(null)}
          snapToGrid
          snapGrid={GRID}
          fitView
          fitViewOptions={{ padding: 0.25, maxZoom: 0.85 }}
          minZoom={0.2}
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

      {codeModal && <CodeModal result={codeModal} onClose={() => setCodeModal(null)} />}
    </div>
  );
}
