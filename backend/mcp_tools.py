"""MCP tool surface — thin adapters over ``ArchitectureStore`` (spec §9).

Each of the 9 tools calls exactly **one** store method and returns its dict.
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


def build_mcp(store, broadcast=None) -> FastMCP:
    """Construct the FastMCP server and register the 9 tools against ``store``.

    ``broadcast`` is an async, no-arg callable invoked after each successful
    mutation; defaults to a no-op so Stage 3 can run before the websocket exists.
    """
    fire = broadcast or _noop_broadcast
    mcp = FastMCP("nn-architect")

    # -- mutations (broadcast on success) -------------------------------------

    @mcp.tool()
    async def add_layer(type: str, params: dict[str, Any]) -> dict[str, Any]:
        """Add a layer node. Validates ``type`` against the catalog and ``params``
        against that type's required/optional fields, applying defaults for any
        omitted. Rejects a second ``input`` node. Returns ``{node_id}``."""
        result = await store.add_layer(type, params)
        await fire()
        return result

    @mcp.tool()
    async def update_layer(node_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Partially update a node's params: merges ``params`` into the existing
        ones and re-validates the merged result. Returns ``{node_id, params}``."""
        result = await store.update_layer(node_id, params)
        await fire()
        return result

    @mcp.tool()
    async def remove_layer(node_id: str) -> dict[str, Any]:
        """Remove a node and cascade-delete every edge touching it.
        Returns ``{removed: true, edges_removed: [...]}``."""
        result = await store.remove_layer(node_id)
        await fire()
        return result

    @mcp.tool()
    async def connect_layers(from_id: str, to_id: str) -> dict[str, Any]:
        """Add a directed edge ``from_id -> to_id``. Rejects: missing id,
        self-loop, ``to_id`` is the input node, duplicate edge, a second incoming
        edge into a non-merge node, or an edge that would create a cycle.
        Returns ``{edge: {from, to}}``."""
        result = await store.connect_layers(from_id, to_id)
        await fire()
        return result

    @mcp.tool()
    async def disconnect_layers(from_id: str, to_id: str) -> dict[str, Any]:
        """Remove the edge ``from_id -> to_id``. Errors if it does not exist.
        Returns ``{removed: true}``."""
        result = await store.disconnect_layers(from_id, to_id)
        await fire()
        return result

    @mcp.tool()
    async def reset_architecture() -> dict[str, Any]:
        """Clear all nodes and edges and reset the id counter so the next id is
        ``n1``. Returns ``{reset: true}``."""
        result = await store.reset_architecture()
        await fire()
        return result

    # -- reads (never broadcast) ----------------------------------------------

    @mcp.tool()
    async def get_architecture() -> dict[str, Any]:
        """Return the full current graph: ``{nodes: [...], edges: [...]}`` with
        edges in stored (insertion) order."""
        return await store.get_architecture()

    @mcp.tool()
    async def validate_architecture() -> dict[str, Any]:
        """Validate the graph by **really executing** generated PyTorch against a
        dummy batch (N=2). On success: ``{valid: true, output_shapes,
        output_node_ids, warnings}``. On failure: ``{valid: false, error:
        {node_id, layer_type, message}, warnings}``."""
        return await store.validate_architecture()

    @mcp.tool()
    async def generate_code() -> dict[str, Any]:
        """Emit best-effort, self-contained PyTorch source for the current graph
        (no prior validate required). Returns ``{code: str}``."""
        return await store.generate_code()

    return mcp
