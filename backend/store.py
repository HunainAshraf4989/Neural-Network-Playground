"""ArchitectureStore: the single source of truth for the graph.

Every mutation of the architecture flows through one of this class's methods.
The MCP tool handlers (``mcp_tools.py``) and the websocket handler (``ws_app.py``)
are thin adapters that call these same methods - there is never a second code
path that mutates the graph, so the canvas can never desync from what the agent
believes exists.

This layer enforces **structural** rules only (unknown type, missing param,
self-loop, cycle, duplicate edge, second input, ``to_id`` is input). Tensor-shape
correctness is decided **exclusively** by real execution in ``validator.py``,
never here. Structural violations are surfaced as ``ValueError``; the adapters
turn those into MCP/websocket error responses.

Node schema is the §7 data model: ``{"id", "type", "params"}`` - geometry never
lives inside a node, so the validator and codegen never see it. The store does
keep an *optional, separate* ``layout`` map (id -> {col, row}) that an agent may
set to hint placement; it is broadcast alongside the graph but kept out of node
params, so it can't affect validation or generated code (relaxed invariant 8).
"""

import asyncio
import copy

import codegen
import layers
import validator


# --------------------------------------------------------------------------- #
# Ungroup blueprints
#
# The persistent counterpart of the frontend's read-only "Expand" view
# (frontend/src/expansions.js): each blueprint describes how a composite layer
# decomposes into REAL catalog types + internal wiring, as name-keyed specs the
# store turns into real nodes (fresh ids). Kept in lock-step with expansions.js
# so what Expand *shows* is exactly what Ungroup *creates*. Plain
# multihead_attention is deliberately absent: its expansion needs a synthetic
# scaled_dot_product_attention type that isn't in the catalog (yet).
#
# A blueprint is ``(specs, edges, entries, exit)``:
#   specs   - [(name, type, params)] in creation order
#   edges   - [(from_name, to_name)] internal wiring
#   entries - names every external predecessor is rewired into
#   exit    - name whose output feeds every external successor
# --------------------------------------------------------------------------- #

def _transformer_blueprint(params):
    """nn.TransformerEncoderLayer (post-norm default, activation=relu), exactly
    as codegen emits it:
      x = norm1(x + dropout(self_attn(x, x, x)))
      x = norm2(x + dropout(linear2(dropout(relu(linear1(x))))))
    The block input feeds self-attn AND the attention-skip residual.
    """
    d, p = params["d_model"], params["dropout"]
    ff = params["dim_feedforward"]
    specs = [
        ("attn", "multihead_attention",
         {"embed_dim": d, "num_heads": params["nhead"], "dropout": p}),
        ("do1", "dropout", {"p": p}),
        ("res1", "add", {}),
        ("norm1", "layernorm", {"normalized_shape": d}),
        ("fc1", "linear", {"in_features": d, "out_features": ff}),
        ("act", "relu", {}),
        ("do2", "dropout", {"p": p}),
        ("fc2", "linear", {"in_features": ff, "out_features": d}),
        ("do3", "dropout", {"p": p}),
        ("res2", "add", {}),
        ("norm2", "layernorm", {"normalized_shape": d}),
    ]
    edges = [
        ("attn", "do1"), ("do1", "res1"), ("res1", "norm1"),
        ("norm1", "fc1"), ("fc1", "act"), ("act", "do2"), ("do2", "fc2"),
        ("fc2", "do3"), ("do3", "res2"), ("res2", "norm2"),
        ("norm1", "res2"),  # feed-forward residual skip
    ]
    return specs, edges, ["attn", "res1"], "norm2"


def _recurrent_blueprint(layer_type, params):
    """A stacked/bidirectional rnn|lstm|gru unrolls into its per-layer cells
    (bidirectional: forward + backward cell -> concat on the feature dim).
    Returns ``None`` for a plain 1-layer unidirectional cell - nothing
    graph-level to unroll (gate internals are below this canvas's altitude).
    """
    num_layers = params["num_layers"]
    bidi = bool(params["bidirectional"])
    if num_layers <= 1 and not bidi:
        return None
    h = params["hidden_size"]
    specs, edges = [], []
    entries, prev_exit = None, None
    for i in range(num_layers):
        in_size = params["input_size"] if i == 0 else h * (2 if bidi else 1)
        cell_params = {"input_size": in_size, "hidden_size": h,
                       "num_layers": 1, "bidirectional": False}
        if bidi:
            fwd, bwd, cat = f"l{i}f", f"l{i}b", f"l{i}c"
            specs += [(fwd, layer_type, dict(cell_params)),
                      (bwd, layer_type, dict(cell_params)),
                      (cat, "concat", {"dim": -1})]
            edges += [(fwd, cat), (bwd, cat)]
            layer_entries, layer_exit = [fwd, bwd], cat
        else:
            name = f"l{i}"
            specs.append((name, layer_type, cell_params))
            layer_entries, layer_exit = [name], name
        if prev_exit is not None:
            edges += [(prev_exit, t) for t in layer_entries]
        if entries is None:
            entries = layer_entries
        prev_exit = layer_exit
    return specs, edges, entries, prev_exit


class ArchitectureStore:
    def __init__(self):
        self.nodes = []          # list[{"id", "type", "params"}], creation order
        self.edges = []          # list[{"from", "to"}], insertion order (matters for concat)
        self.layout = {}         # id -> {col?, row?}: optional frontend placement hints
        self._id_counter = 0     # n1, n2, ...; reset to 0 on reset_architecture
        self._lock = asyncio.Lock()

    # -- internal helpers ---------------------------------------------------

    def _next_id(self):
        self._id_counter += 1
        return f"n{self._id_counter}"

    @staticmethod
    def _clean_layout(layout):
        """Validate an optional placement hint and return the sanitised dict (only
        non-negative integer ``col``/``row`` keys). ``None``/``{}`` -> ``{}``.
        Geometry never enters node params - this is a separate hint channel.
        """
        if layout is None:
            return {}
        if not isinstance(layout, dict):
            raise ValueError(
                f"layout must be an object with optional integer 'col'/'row', "
                f"got {type(layout).__name__}")
        unknown = sorted(set(layout) - {"col", "row"})
        if unknown:
            raise ValueError(f"layout got unknown key(s): {unknown}; allowed: 'col', 'row'")
        cleaned = {}
        for key in ("col", "row"):
            if layout.get(key) is None:
                continue
            value = layout[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"layout '{key}' must be a non-negative integer, got {value!r}")
            cleaned[key] = value
        return cleaned

    def _node(self, node_id):
        return next((n for n in self.nodes if n["id"] == node_id), None)

    def _snapshot(self):
        """Deep copy of the shared state, safe to hand to engine/callers."""
        return {"nodes": copy.deepcopy(self.nodes), "edges": copy.deepcopy(self.edges)}

    def _would_create_cycle(self, from_id, to_id):
        """Adding ``from_id -> to_id`` creates a cycle iff ``from_id`` is already
        reachable from ``to_id`` along existing edges (DFS from target to source).
        """
        adj = {}
        for e in self.edges:
            adj.setdefault(e["from"], []).append(e["to"])
        stack, seen = [to_id], set()
        while stack:
            n = stack.pop()
            if n == from_id:
                return True
            if n in seen:
                continue
            seen.add(n)
            stack.extend(adj.get(n, []))
        return False

    # -- the 9 operations ---------------------------------------------------

    def _add_one(self, layer_type, params, layout=None):
        """Validate and append one node (caller holds the lock). Returns its id.
        Used by both ``add_layer`` and the batched ``add_layers``."""
        if not isinstance(params, dict):
            raise ValueError(f"params must be an object, got {type(params).__name__}")
        if layer_type == "input" and any(n["type"] == "input" for n in self.nodes):
            raise ValueError(
                "an input node already exists; remove_layer the existing one "
                "first to redefine the input")
        merged = layers.validate_and_merge(layer_type, params)
        hint = self._clean_layout(layout)
        node_id = self._next_id()
        self.nodes.append({"id": node_id, "type": layer_type, "params": merged})
        if hint:
            self.layout[node_id] = hint
        return node_id

    async def add_layer(self, layer_type, params, layout=None):
        """Add a node. Validates type+params against the catalog (applying
        defaults). Rejects a second ``input`` node. ``layout`` is an optional
        ``{col, row}`` placement hint stored outside node params.
        """
        async with self._lock:
            return {"node_id": self._add_one(layer_type, params, layout)}

    async def add_layers(self, specs):
        """Add many nodes in one **atomic** call. ``specs`` is a list of
        ``{type, params, layout?}``. Either every node is added (returning ids in
        order) or, on the first invalid spec, nothing is - the graph is rolled
        back so a partial batch never lands. Returns ``{node_ids}``.
        """
        if not isinstance(specs, list) or not specs:
            raise ValueError("add_layers expects a non-empty list of layer specs")
        async with self._lock:
            base_count, base_counter = len(self.nodes), self._id_counter
            added = []
            try:
                for i, spec in enumerate(specs):
                    if not isinstance(spec, dict) or "type" not in spec:
                        raise ValueError(f"spec must be an object with a 'type' key, got {spec!r}")
                    added.append(self._add_one(
                        spec["type"], spec.get("params", {}), spec.get("layout")))
            except ValueError as err:
                del self.nodes[base_count:]          # roll back appended nodes
                self._id_counter = base_counter      # and the ids they consumed
                for nid in added:
                    self.layout.pop(nid, None)
                raise ValueError(f"add_layers failed at index {len(added)}: {err}") from err
            return {"node_ids": added}

    async def update_layer(self, node_id, params):
        """Merge ``params`` into the node's existing params (partial update) and
        re-validate the merged result against the type's schema.
        """
        async with self._lock:
            node = self._node(node_id)
            if node is None:
                raise ValueError(f"unknown node '{node_id}'")
            if not isinstance(params, dict):
                raise ValueError(f"params must be an object, got {type(params).__name__}")
            merged = layers.validate_and_merge(node["type"], {**node["params"], **params})
            node["params"] = merged
            return {"node_id": node_id, "params": copy.deepcopy(merged)}

    async def remove_layer(self, node_id):
        """Remove a node and cascade-delete every edge touching it."""
        async with self._lock:
            node = self._node(node_id)
            if node is None:
                raise ValueError(f"unknown node '{node_id}'")
            removed_edges = [e for e in self.edges if node_id in (e["from"], e["to"])]
            self.edges = [e for e in self.edges if node_id not in (e["from"], e["to"])]
            self.nodes = [n for n in self.nodes if n["id"] != node_id]
            self.layout.pop(node_id, None)
            return {"removed": True, "edges_removed": removed_edges}

    def _connect_one(self, from_id, to_id):
        """Validate and append one edge (caller holds the lock). Returns the edge
        dict. Used by both ``connect_layers`` and ``connect_layers_batch``."""
        if self._node(from_id) is None:
            raise ValueError(f"unknown source node '{from_id}'")
        to_node = self._node(to_id)
        if to_node is None:
            raise ValueError(f"unknown target node '{to_id}'")
        if from_id == to_id:
            raise ValueError(f"self-loop rejected: '{from_id}' cannot connect to itself")
        if to_node["type"] == "input":
            raise ValueError(f"'{to_id}' is the input node and cannot have an incoming edge")
        if any(e["from"] == from_id and e["to"] == to_id for e in self.edges):
            raise ValueError(f"edge '{from_id}' -> '{to_id}' already exists")
        if (to_node["type"] not in layers.MERGE_TYPES
                and any(e["to"] == to_id for e in self.edges)):
            raise ValueError(
                f"node '{to_id}' (type '{to_node['type']}') already has an incoming edge; "
                "non-merge nodes take exactly one input - disconnect it first, or use an "
                "'add'/'concat' merge node to combine multiple inputs")
        if self._would_create_cycle(from_id, to_id):
            raise ValueError(f"edge '{from_id}' -> '{to_id}' would create a cycle")
        edge = {"from": from_id, "to": to_id}
        self.edges.append(edge)
        return dict(edge)

    async def connect_layers(self, from_id, to_id):
        """Add a directed edge. Rejects: missing id, self-loop, ``to_id`` is the
        input node, duplicate edge, or an edge that would create a cycle.
        """
        async with self._lock:
            return {"edge": self._connect_one(from_id, to_id)}

    async def connect_layers_batch(self, edges):
        """Add many edges in one **atomic** call. ``edges`` is a list of
        ``{from_id, to_id}``. Each is validated against the graph *including*
        earlier edges in the same batch; on the first rejection the whole batch
        is rolled back. Returns ``{edges}`` in order.
        """
        if not isinstance(edges, list) or not edges:
            raise ValueError("connect_layers_batch expects a non-empty list of {from_id, to_id}")
        async with self._lock:
            saved = list(self.edges)
            made = []
            try:
                for i, e in enumerate(edges):
                    if not isinstance(e, dict) or "from_id" not in e or "to_id" not in e:
                        raise ValueError(f"edge must be an object with 'from_id'/'to_id', got {e!r}")
                    made.append(self._connect_one(e["from_id"], e["to_id"]))
            except ValueError as err:
                self.edges = saved                   # roll back every edge in this batch
                raise ValueError(
                    f"connect_layers_batch failed at index {len(made)}: {err}") from err
            return {"edges": made}

    async def disconnect_layers(self, from_id, to_id):
        """Remove an existing edge. Errors if the edge does not exist."""
        async with self._lock:
            match = next((e for e in self.edges
                          if e["from"] == from_id and e["to"] == to_id), None)
            if match is None:
                raise ValueError(f"edge '{from_id}' -> '{to_id}' does not exist")
            self.edges.remove(match)
            return {"removed": True}

    async def ungroup(self, node_id):
        """Replace a composite node with its real, editable internal sub-layers
        (the persistent counterpart of the frontend's read-only Expand view).
        Sub-nodes get fresh store ids; internal edges are wired per the type's
        blueprint; external predecessors are rewired into the blueprint's
        ``entries`` and its ``exit`` into the external successors; finally the
        composite is removed. All-or-nothing: any rejection rolls back.

        Ungroupable: ``transformer_encoder_layer``, and ``rnn``/``lstm``/``gru``
        with ``num_layers > 1`` or ``bidirectional=true``.
        Returns ``{ungrouped, node_ids}`` with ``node_ids`` mapping each
        sub-layer's blueprint name (e.g. ``attn``, ``fc1``) to its new id.
        """
        async with self._lock:
            node = self._node(node_id)
            if node is None:
                raise ValueError(f"unknown node '{node_id}'")
            layer_type = node["type"]
            if layer_type == "transformer_encoder_layer":
                blueprint = _transformer_blueprint(node["params"])
            elif layer_type in ("rnn", "lstm", "gru"):
                blueprint = _recurrent_blueprint(layer_type, node["params"])
                if blueprint is None:
                    raise ValueError(
                        f"cannot ungroup '{node_id}': a 1-layer unidirectional "
                        f"{layer_type} has no internal structure to unroll - only "
                        "num_layers > 1 or bidirectional=true decomposes")
            else:
                raise ValueError(
                    f"cannot ungroup '{node_id}' (type '{layer_type}'); only "
                    "transformer_encoder_layer and stacked/bidirectional "
                    "rnn/lstm/gru can be ungrouped")
            specs, internal_edges, entries, exit_name = blueprint

            # _add_one/_connect_one only ever append and node dicts are never
            # mutated here, so shallow list/dict copies are a full rollback.
            saved_nodes, saved_edges = list(self.nodes), list(self.edges)
            saved_layout, saved_counter = dict(self.layout), self._id_counter
            try:
                ids = {name: self._add_one(t, params) for name, t, params in specs}
                for a, b in internal_edges:
                    self._connect_one(ids[a], ids[b])
                # Rewire the composite's external edges IN PLACE (each pred feeds
                # every entry; the exit feeds each successor). In-place keeps edge
                # insertion order, which decides argument order for a downstream
                # concat - remove+append would silently reorder it.
                rewired = []
                for e in self.edges:
                    if e["to"] == node_id:
                        rewired.extend({"from": e["from"], "to": ids[t]} for t in entries)
                    elif e["from"] == node_id:
                        rewired.append({"from": ids[exit_name], "to": e["to"]})
                    else:
                        rewired.append(e)
                self.edges = rewired
                self.nodes = [n for n in self.nodes if n["id"] != node_id]
                self.layout.pop(node_id, None)
            except ValueError:
                self.nodes, self.edges = saved_nodes, saved_edges
                self.layout, self._id_counter = saved_layout, saved_counter
                raise
            return {"ungrouped": node_id,
                    "node_ids": {name: ids[name] for name, _t, _p in specs}}

    async def get_catalog(self):
        """Return the layer catalog (types, categories, required + optional params
        with defaults). Read-only and static, so it needs no lock. This is the
        agent's discovery surface - no probing required."""
        return layers.describe_catalog()

    async def get_architecture(self):
        """Full current state - ``{nodes, edges, layout}``, edges in stored
        (insertion) order. Deep-copied so callers can't mutate shared state by
        reference. ``layout`` carries the optional placement hints (id -> {col,
        row}); it is *not* part of the node schema and never reaches the validator
        or codegen, which read only nodes/edges via ``_snapshot``.
        """
        async with self._lock:
            snap = self._snapshot()
            snap["layout"] = copy.deepcopy(self.layout)
            return snap

    async def validate_architecture(self):
        """Validate the current graph by real execution (delegates to
        ``validator.py``). Runs the blocking subprocess off the event loop.
        """
        async with self._lock:
            snap = self._snapshot()
        return await asyncio.to_thread(validator.validate, snap)

    async def generate_code(self):
        """Best-effort standalone PyTorch source for the current graph
        (delegates to ``codegen.py``). Raises ``ValueError`` if no input node.
        """
        async with self._lock:
            snap = self._snapshot()
        code, _terminals, _warnings = codegen.generate(snap)
        return {"code": code}

    async def reset_architecture(self):
        """Clear all nodes/edges and reset the ID counter so the next id is n1."""
        async with self._lock:
            self.nodes = []
            self.edges = []
            self.layout = {}
            self._id_counter = 0
            return {"reset": True}
