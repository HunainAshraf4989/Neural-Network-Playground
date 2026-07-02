// Top-level app: renders the shared architecture on a React Flow canvas and
// turns user actions into §12 client messages. The backend is the single source
// of truth — every mutation is sent, never applied locally; the canvas only ever
// shows what the server broadcasts back (so it can't drift from the agent's view).
//
// Ownership split (the rule that keeps the canvas from desyncing): the backend
// owns *existence* — which nodes and edges there are — and React Flow owns only
// *geometry* (where they sit) and transient UI (hover/selection). React Flow's
// change stream is therefore guarded so it can never remove a node/edge locally
// (see onNodesChange/onEdgesChange + dropRemoveChanges); a deletion only ever
// lands via a backend broadcast.
//
// Layout note (CLAUDE.md invariant 8): node x/y are NOT in the shared schema —
// they're recomputed each broadcast by the deterministic auto-layout. Overrides
// layer on top, highest first: a user drag position, then an agent {col,row}
// hint, then (for a node that just lost its path from input) its current spot, so
// deleting one wire doesn't reshuffle the downstream graph. Nothing about
// positions ever reaches the backend.

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

import LayerNode, { NEURON_CATEGORIES } from "./LayerNode.jsx";
import LayerPalette from "./LayerPalette.jsx";
import ParamsPanel from "./ParamsPanel.jsx";
import DeletableEdge from "./DeletableEdge.jsx";
import NeuronBundleEdge from "./NeuronBundleEdge.jsx";
import CodeModal from "./CodeModal.jsx";
import { useArchitectureSocket } from "./useArchitectureSocket.js";
import { toFlowElements, dropRemoveChanges } from "./flow.js";
import {
  COL_GAP,
  ROW_GAP,
  computeLayout,
  reachableFromInput,
  colOf,
  rowOf,
  cellPos,
  snapToCell,
} from "./layout.js";
import { defaultParams } from "./catalog.js";
import { computeWidths, computeDiameters, computeNeuronCounts } from "./dims.js";
import { expandGraph, isExpandable } from "./expansions.js";
import * as proto from "./protocol.js";

const NODE_TYPES = { layer: LayerNode };
const EDGE_TYPES = { deletable: DeletableEdge, neuronBundle: NeuronBundleEdge };
const IMAGE_BG = "#0f1420"; // matches the dark canvas background
// Dragging snaps onto the column/row grid so nodes stay aligned into clean columns.
const GRID = [COL_GAP, ROW_GAP];
// Upper bound for auto-fit zoom. fitView still zooms OUT to fit a big graph; this
// only caps how far it zooms IN on a small one — kept generous so a 1–3 node net
// renders large enough to actually read (was 0.85, which left small nets tiny).
const FIT_MAX_ZOOM = 1.4;

export default function App() {
  const { arch, connected, notice, send, requestCode } = useArchitectureSocket();
  const [nodes, setNodes, rawOnNodesChange] = useNodesState([]);
  const [edges, setEdges, rawOnEdgesChange] = useEdgesState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [codeModal, setCodeModal] = useState(null); // {code} | {error} | null
  // "Expand architecture": a read-only view that substitutes composite layers
  // (transformer_encoder_layer, multihead_attention, stacked/bidi RNNs) with their
  // internal subgraph. Purely a frontend view transform (expansions.js) — the store
  // never sees the sub-nodes (invariant 8), so it can't desync the canvas.
  const [expanded, setExpanded] = useState(false);
  const rfRef = useRef(null);
  const prevNodeCount = useRef(0);
  const dragOverridesRef = useRef(new Map()); // id -> {x,y}: where the user dragged a node
  const prevIdsRef = useRef(new Set()); // store ids seen last broadcast, to spot a new node
  const pendingDropRef = useRef(null); // {x,y}: where a palette item was just dropped

  // The backend is authoritative for *which* nodes/edges exist; the frontend
  // owns pixel layout. Positions are recomputed every broadcast from the graph
  // itself (`computeLayout`: column = distance from input, so wiring n1->n2 puts
  // n2 one column right of n1), then overrides are layered on top, highest first:
  //   1. a node the user dragged keeps its dragged position;
  //   2. else an agent placement hint ({col,row} on the backend `layout`) wins;
  //   3. else, if the node has *lost* its path from input (e.g. the user deleted
  //      an upstream edge), it stays where it already is rather than collapsing
  //      back to the input column — so deleting one wire doesn't reshuffle the
  //      whole downstream graph. Once it's re-wired, the edge-driven layout takes
  //      over again.
  // Existence is never applied locally: nodes/edges only appear/disappear from a
  // broadcast (see the guarded change handlers below), so the canvas can't drift
  // from the agent's view.
  useEffect(() => {
    // The geometry pipeline (layout/sizing/glyphs) runs on the *view* graph: the
    // store graph itself, or — when "Expand architecture" is on — that graph with
    // composite nodes substituted by their internal subgraph. expandGraph is a pure
    // {nodes,edges,layout} -> {nodes,edges,layout} transform, so everything below is
    // unchanged. Drag-override cleanup still keys off the *store* ids (existence),
    // not the view.
    const graph = expanded ? expandGraph(arch) : arch;
    const ids = new Set((arch?.nodes ?? []).map((n) => n.id));
    const { nodes: n, edges: e } = toFlowElements(graph);

    // Forget drag positions for nodes that no longer exist.
    for (const id of [...dragOverridesRef.current.keys()]) {
      if (!ids.has(id)) dragOverridesRef.current.delete(id);
    }

    // A palette drop remembered where it landed (pendingDropRef). Once the backend
    // echoes the new node back, park it at that spot — the SAME frontend-only
    // mechanism as a manual drag (a drag override), so the drop position never
    // reaches the store (invariant 8). Match by the single id that's new since the
    // last broadcast; if a drop somehow produced no/many new nodes, drop the hint.
    if (pendingDropRef.current) {
      const fresh = [...ids].filter((id) => !prevIdsRef.current.has(id));
      if (fresh.length === 1) {
        dragOverridesRef.current.set(fresh[0], pendingDropRef.current);
        pendingDropRef.current = null;
      } else if (fresh.length > 1) {
        pendingDropRef.current = null;
      }
    }
    prevIdsRef.current = ids;

    const base = computeLayout(graph); // edge-driven columns for every node
    const hints = graph?.layout ?? {}; // agent placement hints (id -> {col,row})
    const reachable = reachableFromInput(graph); // ids with a path from input
    const hasInput = (graph?.nodes ?? []).some((node) => node.type === "input");
    // Glyph size + label per node from its feature width (dims.js). Frontend-only,
    // exactly like positions — it never reaches the backend (invariant 8).
    const diameters = computeDiameters(graph);
    const widths = computeWidths(graph);
    // Representative neuron count per node — drawn only by the linear glyph so the
    // number of neurons on screen reads as the layer's size (see LayerNode).
    const neurons = computeNeuronCounts(graph);

    // Functional updater so we can read the *current* on-screen nodes without
    // adding `nodes` to the deps (which would re-run mid-drag and snap nodes
    // back). Reusing each existing node object also preserves React Flow's
    // internal measurements, so edges never blink out during a re-measure.
    setNodes((prevNodes) => {
      const prevById = new Map(prevNodes.map((p) => [p.id, p]));
      const positionFor = (id) => {
        if (dragOverridesRef.current.has(id)) return dragOverridesRef.current.get(id);
        const hint = hints[id];
        const b = base[id] ?? { x: 0, y: 0 };
        if (hint && (hint.col != null || hint.row != null)) {
          return cellPos(hint.col != null ? hint.col : colOf(b), hint.row != null ? hint.row : rowOf(b));
        }
        // Node that dropped out of the reachable set keeps its current spot.
        if (hasInput && !reachable.has(id) && prevById.has(id)) {
          return prevById.get(id).position;
        }
        return b;
      };
      return n.map((node) => {
        const prev = prevById.get(node.id);
        const position = positionFor(node.id);
        const data = { ...node.data, diameter: diameters[node.id], width: widths[node.id], neurons: neurons[node.id], expanded };
        // Expansion sub-nodes don't exist in the store, so they're inert: React
        // Flow may place them but never drag/select/connect/delete them.
        const flags = data.synthetic
          ? { draggable: false, selectable: false, connectable: false, deletable: false }
          : { draggable: true, selectable: true, connectable: true, deletable: true };
        return prev
          ? { ...prev, type: node.type, data, position, ...flags }
          : { ...node, data, position, ...flags };
      });
    });
    // In the expanded view, an edge between two neuron columns (input/linear/
    // activation, non-synthetic) is drawn as the fully-connected bundle of a classic
    // deep-net diagram instead of a single line (see NeuronBundleEdge).
    const neuronIds = expanded
      ? new Set(
          n
            .filter((node) => !node.data.synthetic && NEURON_CATEGORIES.has(node.data.category))
            .map((node) => node.id),
        )
      : new Set();
    setEdges(
      e.map((edge) => {
        if (neuronIds.has(edge.source) && neuronIds.has(edge.target)) {
          return {
            ...edge,
            type: "neuronBundle",
            data: {
              sourceCount: neurons[edge.source] ?? 3,
              targetCount: neurons[edge.target] ?? 3,
              // A column whose real width exceeds its drawn circles shows a ⋮ in the
              // middle slot; the bundle must skip that slot too (see NeuronBundleEdge).
              sourceTruncated: (widths[edge.source] ?? 0) > (neurons[edge.source] ?? 3),
              targetTruncated: (widths[edge.target] ?? 0) > (neurons[edge.target] ?? 3),
            },
          };
        }
        return {
          ...edge,
          type: "deletable",
          // Synthetic (expansion) edges aren't store edges, so they can't be deleted;
          // a `null` onDelete hides the hover ✕ (see DeletableEdge).
          data: {
            onDelete: edge.source.includes("/") || edge.target.includes("/")
              ? null
              : () => send(proto.disconnectLayers(edge.source, edge.target)),
          },
        };
      }),
    );
  }, [arch, expanded, setNodes, setEdges, send]);

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
        rfRef.current?.fitView({ padding: 0.25, maxZoom: FIT_MAX_ZOOM, duration: 300 }),
      );
    }
  }, [arch]);

  // Toggling expand changes the node set wholesale, so re-fit the view to it.
  useEffect(() => {
    if (rfRef.current) {
      requestAnimationFrame(() =>
        rfRef.current?.fitView({ padding: 0.2, maxZoom: FIT_MAX_ZOOM, duration: 300 }),
      );
    }
  }, [expanded]);

  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedId) ?? null,
    [nodes, selectedId],
  );

  // Whether Expand would actually change the view: a composite block that unfolds
  // into a subgraph (isExpandable), OR a linear layer, which expands into its neuron
  // box. A plain 1-layer RNN with no linears has nothing to show, so the button
  // disables itself.
  const hasExpandable = useMemo(
    () => (arch?.nodes ?? []).some((n) => isExpandable(n) || n.type === "linear"),
    [arch],
  );

  const add = useCallback((type) => send(proto.addLayer(type, defaultParams(type))), [send]);

  // The expanded view is read-only (its sub-nodes aren't in the store), so wiring
  // is disabled while it's on — toggle off to edit the real graph.
  const onConnect = useCallback(
    (c) => {
      if (expanded) return;
      send(proto.connectLayers(c.source, c.target));
    },
    [expanded, send],
  );
  const onNodesDelete = useCallback(
    (deleted) => deleted.forEach((n) => send(proto.removeLayer(n.id))),
    [send],
  );
  const onEdgesDelete = useCallback(
    (deleted) => deleted.forEach((e) => send(proto.disconnectLayers(e.source, e.target))),
    [send],
  );

  // React Flow may freely move/measure/select on the canvas, but it must never
  // *delete* a node/edge from our controlled state on its own — existence lives
  // in the backend. We drop `remove` changes (a known source of the canvas
  // dropping edges it can't momentarily resolve); a real user delete still fires
  // onNodesDelete/onEdgesDelete -> backend -> broadcast. See dropRemoveChanges.
  const onNodesChange = useCallback(
    (changes) => rawOnNodesChange(dropRemoveChanges(changes)),
    [rawOnNodesChange],
  );
  const onEdgesChange = useCallback(
    (changes) => rawOnEdgesChange(dropRemoveChanges(changes)),
    [rawOnEdgesChange],
  );

  // Remember where the user dragged a node (snapped to the grid) so it overrides
  // the edge-driven layout and stays put across broadcasts.
  const onNodeDragStop = useCallback((_, node) => {
    dragOverridesRef.current.set(node.id, snapToCell(node.position));
  }, []);

  // Dropping a palette item adds the layer and remembers where it was dropped, so
  // the new node lands in that column instead of the default column 0 (which the
  // edge-driven layout would otherwise use until it's wired).
  const onDrop = useCallback(
    (event) => {
      event.preventDefault();
      const type = event.dataTransfer.getData("application/nn-layer-type");
      if (!type) return;
      const flowPos = rfRef.current?.screenToFlowPosition?.({ x: event.clientX, y: event.clientY });
      pendingDropRef.current = flowPos ? snapToCell(flowPos) : null;
      add(type);
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
          <button
            type="button"
            className={`toolbar__btn${expanded ? " toolbar__btn--active" : ""}`}
            onClick={() => {
              setExpanded((v) => !v);
              setSelectedId(null);
            }}
            disabled={!hasExpandable && !expanded}
            title="Expand: unfold composite blocks (transformers, attention, stacked RNNs) into their sub-layers, and show linear layers as their neurons"
          >
            {expanded ? "Collapse" : "Expand"}
          </button>
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
          onNodeClick={(_, node) => {
            if (node.data?.synthetic) return; // sub-nodes have no editable store entry
            setSelectedId(node.id);
          }}
          onPaneClick={() => setSelectedId(null)}
          snapToGrid
          snapGrid={GRID}
          fitView
          fitViewOptions={{ padding: 0.25, maxZoom: FIT_MAX_ZOOM }}
          minZoom={0.2}
        >
          <Background />
          <Controls />
        </ReactFlow>

        {/* Empty-canvas guidance so a blank screen is never a mystery: when not
            connected, point the user at starting the backend (the canvas
            reconnects on its own); when connected, invite the first layer.
            Hidden the moment any node exists, so it never covers a real graph. */}
        {nodes.length === 0 && (
          <div className="canvas__empty">
            {connected ? (
              <>
                <div className="canvas__empty-title">Canvas is live</div>
                <p>
                  Drag a layer from the palette, or ask your agent to build a
                  network — it appears here as it's created.
                </p>
              </>
            ) : (
              <>
                <div className="canvas__empty-title">Waiting for the backend…</div>
                <p>
                  The canvas reconnects automatically. Make sure a backend is
                  running — Claude Desktop launches it, or run{" "}
                  <code>MODE=standalone python backend/main.py</code>.
                </p>
              </>
            )}
          </div>
        )}
      </div>

      <ParamsPanel
        node={selectedNode}
        onSave={(id, params) => send(proto.updateLayer(id, params))}
        onDelete={(id) => {
          send(proto.removeLayer(id));
          setSelectedId(null);
        }}
        onUngroup={(id) => {
          // The composite is replaced by its sub-layers on the next broadcast,
          // so the selection would point at a node that no longer exists.
          send(proto.ungroup(id));
          setSelectedId(null);
        }}
        onClose={() => setSelectedId(null)}
      />

      {codeModal && <CodeModal result={codeModal} onClose={() => setCodeModal(null)} />}
    </div>
  );
}
