"""ArchitectureStore: the single source of truth for the graph.

Every mutation of the architecture flows through one of this class's methods.
The MCP tool handlers (``mcp_tools.py``) and the websocket handler (``ws_app.py``)
are thin adapters that call these same methods — there is never a second code
path that mutates the graph, so the canvas can never desync from what the agent
believes exists.

This layer enforces **structural** rules only (unknown type, missing param,
self-loop, cycle, duplicate edge, second input, ``to_id`` is input). Tensor-shape
correctness is decided **exclusively** by real execution in ``validator.py`` —
never here. Structural violations are surfaced as ``ValueError``; the adapters
turn those into MCP/websocket error responses.

Node schema is the §7 data model: ``{"id", "type", "params"}``. Positions are NOT
stored — layout is a frontend concern, recomputed on every state push.
"""

import asyncio
import copy

import codegen
import layers
import validator


class ArchitectureStore:
    def __init__(self):
        self.nodes = []          # list[{"id", "type", "params"}], creation order
        self.edges = []          # list[{"from", "to"}], insertion order (matters for concat)
        self._id_counter = 0     # n1, n2, ...; reset to 0 on reset_architecture
        self._lock = asyncio.Lock()

    # -- internal helpers ---------------------------------------------------

    def _next_id(self):
        self._id_counter += 1
        return f"n{self._id_counter}"

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

    async def add_layer(self, layer_type, params):
        """Add a node. Validates type+params against the catalog (applying
        defaults). Rejects a second ``input`` node.
        """
        async with self._lock:
            if layer_type == "input" and any(n["type"] == "input" for n in self.nodes):
                raise ValueError(
                    "an input node already exists; remove_layer the existing one "
                    "first to redefine the input")
            merged = layers.validate_and_merge(layer_type, params)
            node_id = self._next_id()
            self.nodes.append({"id": node_id, "type": layer_type, "params": merged})
            return {"node_id": node_id}

    async def update_layer(self, node_id, params):
        """Merge ``params`` into the node's existing params (partial update) and
        re-validate the merged result against the type's schema.
        """
        async with self._lock:
            node = self._node(node_id)
            if node is None:
                raise ValueError(f"unknown node '{node_id}'")
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
            return {"removed": True, "edges_removed": removed_edges}

    async def connect_layers(self, from_id, to_id):
        """Add a directed edge. Rejects: missing id, self-loop, ``to_id`` is the
        input node, duplicate edge, or an edge that would create a cycle.
        """
        async with self._lock:
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
                    "non-merge nodes take exactly one input — disconnect it first, or use an "
                    "'add'/'concat' merge node to combine multiple inputs")
            if self._would_create_cycle(from_id, to_id):
                raise ValueError(f"edge '{from_id}' -> '{to_id}' would create a cycle")
            edge = {"from": from_id, "to": to_id}
            self.edges.append(edge)
            return {"edge": dict(edge)}

    async def disconnect_layers(self, from_id, to_id):
        """Remove an existing edge. Errors if the edge does not exist."""
        async with self._lock:
            match = next((e for e in self.edges
                          if e["from"] == from_id and e["to"] == to_id), None)
            if match is None:
                raise ValueError(f"edge '{from_id}' -> '{to_id}' does not exist")
            self.edges.remove(match)
            return {"removed": True}

    async def get_architecture(self):
        """Full current state, edges in stored (insertion) order. Deep-copied so
        callers can't mutate the shared state by reference.
        """
        async with self._lock:
            return self._snapshot()

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
            self._id_counter = 0
            return {"reset": True}
