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
  back to source), `to_id` is `input`, exact edge already exists (reject duplicate, don't no-op), or
  `to_id` is a non-merge node that already has an incoming edge (§8: non-input/non-merge nodes take
  **exactly one** input; only `add`/`concat` accept ≥2). The duplicate check runs before this one;
  this one runs before the cycle check.
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
✅ done. `backend/store.py` wraps the Stage 1 engine with the 9 async operations, an `n1/n2…`
counter, and an `asyncio.Lock` guarding every method. `backend/tests/test_store.py` has 31 tests —
all green via `/home/hunain/base/bin/python -m pytest backend/tests/test_store.py -q`. Coverage:
happy path + every rejection branch for all 9 ops (including the "exactly one inbound for non-merge"
guard, and that merge nodes still accept ≥2 inbound), edge-insertion-order preservation (concat),
deep-copy isolation of `get_architecture`, counter reset, orphan-node tolerance, and real-execution
validation (CNN passes → `[[2,10]]`; channel mismatch → `valid:false` naming `n3`/`conv2d`).

Notes / deviations:
- **All 9 methods are `async`** (not just mutators) so MCP/ws adapters can `await` them uniformly;
  each acquires `self._lock`. Reads (`get`/`validate`/`generate`) take a deep-copied snapshot under
  the lock, then operate outside it — `validate_architecture` runs the blocking subprocess via
  `asyncio.to_thread` so a 10s validation can't freeze the event loop serving websockets.
- **Errors are raised as `ValueError`** with clear messages; the adapters (Stage 3/4) translate them
  into MCP/websocket error responses. No error codes/enums — kept minimal.
- `get_architecture` returns the pure §7 schema (`id`/`type`/`params`); `category` is **not** surfaced
  yet — it has no consumer until the websocket/frontend stages, so it's deferred there (Delta 2).
- `generate_code` returns `{"code": str}` per §9 and propagates `ValueError("graph has no input
  node")` from codegen when there's no input.
- Tests are dependency-free (no `pytest-asyncio`): an `@asynctest` shim wraps each `async def` in
  `asyncio.run`, and the store is built *inside* the coroutine so its `Lock` binds to that loop.
- `pytest` added to `backend/requirements.txt` (the gate now depends on it).
- Latent (out of scope, Stage 1 design): `validator.py` writes a fixed shared path
  `/tmp/_nn_architect_generated.py`, so concurrent validations can race. Per-call validation is fine;
  flagging for if/when concurrency matters.
