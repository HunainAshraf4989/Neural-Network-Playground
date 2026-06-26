# Stage 2 — ArchitectureStore + tests

## Scope
`backend/store.py` — the single source of truth wrapping the Stage 1 engine with the 9 operations and
all **structural** validation/error rules. Plus `backend/tests/test_store.py`.

## The store
- State: `nodes: list[Node]`, `edges: list[Edge]` (insertion order preserved — matters for concat),
  `_id_counter` for `n1/n2…`. Guard mutations with an `asyncio.Lock`.
- Methods (one per MCP tool, called by both `mcp_tools.py` and `ws_app.py`):
  `add_layer`, `update_layer`, `remove_layer` (cascade-deletes touching edges),
  `connect_layers`, `disconnect_layers`, `get_architecture`, `validate_architecture` (delegates to
  `validator.py`), `generate_code` (delegates to `codegen.py`), `reset_architecture`.

## Rules to enforce (structural only — never tensor shapes)
- Exactly one `input` node; reject a second (tell agent to remove the existing one). `input` can't be
  a `to_id`.
- `add_layer`/`update_layer`: validate type against catalog, params against required/optional schema,
  apply defaults; `update_layer` merges partial params then re-validates.
- `connect_layers` errors: missing id, self-loop (`from==to`), would create a cycle (DFS from target
  back to source), `to_id` is `input`, or exact edge already exists (reject duplicate, don't no-op).
- `disconnect_layers`: error if edge absent.
- Orphan nodes (unreachable from `input`) are allowed — not an error.

## Expected outcome
Unit tests cover every rule: happy path for each op + each rejection branch.

## Verification gate
```bash
/home/hunain/base/bin/python -m pytest backend/tests/test_store.py -q
```
All tests pass. (pytest installs into `/home/hunain/base` if missing.)

## Status
⬜ not started.
