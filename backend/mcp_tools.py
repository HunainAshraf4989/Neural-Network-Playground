"""MCP tool surface — thin adapters over ``ArchitectureStore`` (spec §9).

Each tool calls exactly **one** store method and returns its dict.
There is no graph-mutation logic in this module: the store is the single
mutation path (CLAUDE.md invariant 2), so the canvas the human sees can never
drift from what the agent believes exists.

Structural errors are raised by the store as ``ValueError``; FastMCP turns any
exception raised inside a tool into an MCP error result (``isError: true`` with
the message), so the agent always learns *why* a call was rejected. After every
**successful mutation** we fire the ``broadcast`` hook so websocket clients
(Stage 4) re-render — read-only tools never broadcast.

stdout is reserved for MCP protocol framing (CLAUDE.md invariant 1): nothing
here prints; the module logger writes to stderr.
"""

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

log = logging.getLogger(__name__)


async def _noop_broadcast() -> None:
    """Default broadcast hook for Stage 3 — wired to ws_app in Stage 4."""


def build_mcp(store, broadcast=None, canvas_warning=None) -> FastMCP:
    """Construct the FastMCP server and register the tools against ``store``.

    ``broadcast`` is an async, no-arg callable invoked after each successful
    mutation; defaults to a no-op so Stage 3 can run before the websocket exists.

    ``canvas_warning`` is an optional string set by ``main.py`` when the live
    canvas is unavailable (the websocket port was busy, so this process runs
    MCP-only). When set, it is attached to every mutation and ``get_architecture``
    response so the agent — whose ONLY channel is the MCP reply, never stderr —
    learns the UI will not reflect its changes.
    """
    fire = broadcast or _noop_broadcast
    mcp = FastMCP("nn-architect")

    def flag(result):
        """Attach ``canvas_warning`` to a result without mutating the original
        (the store may hand back shared state). No-op when the canvas is up."""
        if canvas_warning and isinstance(result, dict):
            return {**result, "warnings": [*result.get("warnings", []), canvas_warning]}
        return result

    # -- mutations (broadcast on success) -------------------------------------

    @mcp.tool()
    async def add_layer(type: str, params: dict[str, Any],
                        layout: dict[str, Any] | None = None) -> dict[str, Any]:
        """Add a layer node. Validates ``type`` against the catalog and ``params``
        against that type's required/optional fields, applying defaults for any
        omitted. Rejects a second ``input`` node. ``layout`` is an OPTIONAL
        placement hint ``{"col": int, "row": int}`` (both >= 0, either omittable)
        used only by the canvas — it never affects validation or generated code.
        Call ``get_catalog`` first to learn valid types and their params.
        Returns ``{node_id}``."""
        log.info("mcp add_layer type=%s params=%s layout=%s", type, params, layout)
        result = await store.add_layer(type, params, layout)
        await fire()
        return flag(result)

    @mcp.tool()
    async def add_layers(layers: list[dict[str, Any]]) -> dict[str, Any]:
        """Add MANY layers in one atomic call — use this instead of looping
        ``add_layer`` when building a whole network. ``layers`` is a list of
        ``{"type": str, "params": dict, "layout"?: {"col","row"}}``. Either all
        are added (ids returned in order, e.g. ``n5``..``n9``) or, on the first
        invalid spec, none are (the graph is rolled back). Returns ``{node_ids}``."""
        log.info("mcp add_layers count=%s", len(layers) if isinstance(layers, list) else "?")
        result = await store.add_layers(layers)
        await fire()
        return flag(result)

    @mcp.tool()
    async def update_layer(node_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Partially update a node's params: merges ``params`` into the existing
        ones and re-validates the merged result. Returns ``{node_id, params}``."""
        log.info("mcp update_layer node_id=%s params=%s", node_id, params)
        result = await store.update_layer(node_id, params)
        await fire()
        return flag(result)

    @mcp.tool()
    async def remove_layer(node_id: str) -> dict[str, Any]:
        """Remove a node and cascade-delete every edge touching it.
        Returns ``{removed: true, edges_removed: [...]}``."""
        log.info("mcp remove_layer node_id=%s", node_id)
        result = await store.remove_layer(node_id)
        await fire()
        return flag(result)

    @mcp.tool()
    async def connect_layers(from_id: str, to_id: str) -> dict[str, Any]:
        """Add a directed edge ``from_id -> to_id``. Rejects: missing id,
        self-loop, ``to_id`` is the input node, duplicate edge, a second incoming
        edge into a non-merge node, or an edge that would create a cycle.
        Returns ``{edge: {from, to}}``."""
        log.info("mcp connect_layers %s -> %s", from_id, to_id)
        result = await store.connect_layers(from_id, to_id)
        await fire()
        return flag(result)

    @mcp.tool()
    async def connect_layers_batch(edges: list[dict[str, str]]) -> dict[str, Any]:
        """Add MANY edges in one atomic call — use this instead of looping
        ``connect_layers`` when wiring a whole network. ``edges`` is a list of
        ``{"from_id": str, "to_id": str}``, applied in order with the same rules
        as ``connect_layers`` (each checked against earlier edges in the batch).
        On the first rejection the whole batch is rolled back. Returns
        ``{edges}``."""
        log.info("mcp connect_layers_batch count=%s", len(edges) if isinstance(edges, list) else "?")
        result = await store.connect_layers_batch(edges)
        await fire()
        return flag(result)

    @mcp.tool()
    async def disconnect_layers(from_id: str, to_id: str) -> dict[str, Any]:
        """Remove the edge ``from_id -> to_id``. Errors if it does not exist.
        Returns ``{removed: true}``."""
        log.info("mcp disconnect_layers %s -> %s", from_id, to_id)
        result = await store.disconnect_layers(from_id, to_id)
        await fire()
        return flag(result)

    @mcp.tool()
    async def reset_architecture() -> dict[str, Any]:
        """Clear all nodes and edges and reset the id counter so the next id is
        ``n1``. Returns ``{reset: true}``."""
        log.info("mcp reset_architecture")
        result = await store.reset_architecture()
        await fire()
        return flag(result)

    # -- reads (never broadcast) ----------------------------------------------

    @mcp.tool()
    async def get_catalog() -> dict[str, Any]:
        """List every layer type the server understands, each with its
        ``category``, ``required`` params, and ``optional`` params with defaults.
        Call this FIRST so you build with valid type strings and param names
        instead of guessing. Returns ``{type: {category, required, optional}}``."""
        log.info("mcp get_catalog")
        return await store.get_catalog()

    @mcp.tool()
    async def get_architecture() -> dict[str, Any]:
        """Return the full current graph: ``{nodes: [...], edges: [...], layout:
        {...}}`` with edges in stored (insertion) order. ``layout`` holds the
        optional per-node placement hints (id -> {col, row}); it is canvas-only
        and never affects validation or codegen."""
        log.info("mcp get_architecture")
        return flag(await store.get_architecture())

    @mcp.tool()
    async def validate_architecture() -> dict[str, Any]:
        """Validate the graph by **really executing** generated PyTorch against a
        dummy batch (N=2). On success: ``{valid: true, output_shapes,
        output_node_ids, warnings}``. On failure: ``{valid: false, error:
        {node_id, layer_type, message}, warnings}``."""
        log.info("mcp validate_architecture")
        return await store.validate_architecture()

    @mcp.tool()
    async def generate_code() -> dict[str, Any]:
        """Emit best-effort, self-contained PyTorch source for the current graph
        (no prior validate required). Returns ``{code: str}``."""
        log.info("mcp generate_code")
        return await store.generate_code()

    return mcp
