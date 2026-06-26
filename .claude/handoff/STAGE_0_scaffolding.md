# Stage 0 — Scaffolding, CLAUDE.md, handoff docs, deps

## Scope
Stand up the repo skeleton and dependencies so later stages can focus purely on code. No app logic.

## Files
- `backend/requirements.txt` — fastapi, uvicorn[standard], websockets, pydantic, mcp, torch (CPU).
- `.gitignore`, `CLAUDE.md`, `.claude/handoff/STAGE_*.md` (these files).
- Directory tree: `backend/tests/`, `frontend/src/`, `.claude/handoff/`.

## Environment
Deps install into the shared venv at **`/home/hunain/base`** (no project venv, no Docker):
```bash
/home/hunain/base/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
/home/hunain/base/bin/pip install -r backend/requirements.txt
```

## Expected outcome
Directory tree exists; `/home/hunain/base` has torch, mcp, fastapi, uvicorn, websockets, pydantic;
CLAUDE.md + handoff docs present.

## Verification gate
```bash
/home/hunain/base/bin/pip list | grep -Ei 'torch|mcp|fastapi|uvicorn|websockets|pydantic'
/home/hunain/base/bin/python -c "import torch, mcp, fastapi, uvicorn, websockets, pydantic; print('ok')"
```
Both must succeed (`ok` printed, all six deps listed).

## Status
✅ done. `pip list` shows fastapi 0.138.1, mcp 1.28.1, pydantic 2.13.4, torch 2.12.1+cpu, uvicorn
0.49.0, websockets 16.0 (+ numpy added so torch imports cleanly). Import check prints `ok`.
