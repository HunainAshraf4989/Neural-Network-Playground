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
import { COL_GAP, ROW_GAP } from "./layout.js";
import { defaultParams } from "./catalog.js";
import * as proto from "./protocol.js";

const NODE_TYPES = { layer: LayerNode };
const EDGE_TYPES = { deletable: DeletableEdge };
const IMAGE_BG = "#0f1420"; // matches the dark canvas background
// Snap grid aligned to the auto-layout spacing, so snapped nodes line up into the
// same layered columns/rows the deterministic layout produces.
const SNAP_GRID = [COL_GAP / 2, ROW_GAP / 2];

const finitePos = (p) => !!p && Number.isFinite(p.x) && Number.isFinite(p.y);
const snap = (p) => ({
  x: Math.round(p.x / SNAP_GRID[0]) * SNAP_GRID[0],
  y: Math.round(p.y / SNAP_GRID[1]) * SNAP_GRID[1],
});

export default function App() {
  const { arch, connected, notice, send, requestCode } = useArchitectureSocket();
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [codeModal, setCodeModal] = useState(null); // {code} | {error} | null
  const [snapToColumns, setSnapToColumns] = useState(false);
  const rfRef = useRef(null);
  const prevNodeCount = useRef(0);
  const positionsRef = useRef(new Map()); // id -> {x,y} the user dragged/dropped
  const knownIdsRef = useRef(new Set()); // ids seen in the previous broadcast
  const pendingDropRef = useRef(null); // graph-coords for the next dropped node
  const suppressFitRef = useRef(false); // skip auto-fit for a drop placement

  // Server state is authoritative for *which* nodes/edges exist, but positions
  // are frontend-only and **stable**: once a node has a position we never move
  // it on a later broadcast. Only brand-new nodes get a position — from the drop
  // cursor if they were just dropped, else from the deterministic auto-layout.
  // This is what keeps existing nodes from shifting or vanishing when another
  // node is added/dropped (and a bad cursor coord can never poison a node).
  useEffect(() => {
    const ids = new Set((arch?.nodes ?? []).map((n) => n.id));
    const { nodes: n, edges: e } = toFlowElements(arch);

    // Drop the cursor position onto the one genuinely-new node (if a drop is
    // pending and the coord is valid); everyone else seeds from auto-layout.
    const drop =
      pendingDropRef.current && finitePos(pendingDropRef.current)
        ? pendingDropRef.current
        : null;
    pendingDropRef.current = null;
    for (const node of n) {
      if (positionsRef.current.has(node.id)) continue; // already placed — leave it
      const isNew = !knownIdsRef.current.has(node.id);
      const seed = isNew && drop ? drop : node.position;
      positionsRef.current.set(node.id, finitePos(seed) ? seed : { x: 0, y: 0 });
    }
    // Forget positions for nodes that no longer exist.
    for (const id of [...positionsRef.current.keys()]) {
      if (!ids.has(id)) positionsRef.current.delete(id);
    }

    setNodes(n.map((node) => ({ ...node, position: positionsRef.current.get(node.id) })));
    setEdges(
      e.map((edge) => ({
        ...edge,
        type: "deletable",
        data: { onDelete: () => send(proto.disconnectLayers(edge.source, edge.target)) },
      })),
    );
    knownIdsRef.current = ids;
  }, [arch, setNodes, setEdges, send]);

  // Re-frame only when a node is ADDED, so a new node is always visible — but
  // leave the view alone on delete and on param edits (which previously caused
  // the canvas to jump/zoom unexpectedly), and don't fight a deliberate drop
  // placement. `maxZoom` keeps the fit from zooming way in on a one/two-node
  // graph (which made the circles huge).
  useEffect(() => {
    const count = arch?.nodes?.length ?? 0;
    const grew = count > prevNodeCount.current;
    prevNodeCount.current = count;
    if (!grew) return;
    if (suppressFitRef.current) {
      suppressFitRef.current = false;
      return;
    }
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

  // Remember where the user dropped a node so it stays put across broadcasts.
  const onNodeDragStop = useCallback((_, node) => {
    positionsRef.current.set(node.id, node.position);
  }, []);

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();
      const type = event.dataTransfer.getData("application/nn-layer-type");
      if (!type) return;
      if (rfRef.current) {
        // Place the new node exactly where it was dropped (graph coords),
        // snapping it onto the column grid when that mode is on.
        const p = rfRef.current.screenToFlowPosition({ x: event.clientX, y: event.clientY });
        pendingDropRef.current = snapToColumns ? snap(p) : p;
        suppressFitRef.current = true;
      }
      add(type);
    },
    [add, snapToColumns],
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
