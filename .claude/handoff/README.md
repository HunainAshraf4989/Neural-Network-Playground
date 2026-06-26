# Handoff docs

One file per build stage so any agent can pick up mid-build. Each `STAGE_<n>_*.md` states the scope,
files to create, the expected outcome, and the **exact verification gate** that must pass before the
next stage starts. A stage is only "done" when its gate line shows ✅ with evidence.

Canonical references:
- `../../Tentative plan.md` — full build spec (sections §N).
- `~/.claude/plans/iridescent-bubbling-noodle.md` — approved staged plan + the two scope deltas
  (broad catalog, light polish).
- `../../CLAUDE.md` — invariants and environment (deps live in `/home/hunain/base`).

## Status board

| Stage | Title | Gate status |
|---|---|---|
| 0 | Scaffolding, CLAUDE.md, handoff docs, deps | ✅ done |
| 1 | Core engine (layers, codegen, validator)  | ⬜ not started |
| 2 | ArchitectureStore + tests                 | ⬜ not started |
| 3 | MCP tools + main.py MCP loop              | ⬜ not started |
| 4 | WebSocket app + standalone server         | ⬜ not started |
| 5 | React frontend                            | ⬜ not started |
| 6 | Polish, README, portfolio demo            | ⬜ not started |
