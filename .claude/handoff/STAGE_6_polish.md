# Stage 6 — Polish, README, portfolio demo

## Scope
Light demo-ready polish + the portfolio README + Claude Desktop registration. No new architecture.

## Files / work
- Finalize category colors + a cohesive light theme over React Flow defaults; grouped palette;
  readable nodes. No custom animations/branding.
- `README.md` (repo root, portfolio centerpiece): lead with "collaborative agent-driven design +
  live validation" (§1). Include the architecture diagram (§4), a recorded **GIF** of Claude Desktop
  building a CNN live on the canvas, the standalone-codegen demo, setup/run instructions (using
  `/home/hunain/base`), and the catalog overview.
- `claude_desktop_config.snippet.json` — points `command` at `/home/hunain/base/bin/python` and
  `args` at the absolute `backend/main.py` path.

## Verification gate — full acceptance (spec §16 + autoencoder)
Via Claude Desktop, with the canvas open and no manual refresh:
- [ ] Build a small CNN (28×28×1, 10-class) — appears live.
- [ ] Add a residual skip (`add` merge) — codegen + validation handle it.
- [ ] Build a Transformer-encoder block — generated code runs standalone.
- [ ] Build an LSTM sequence model — validation passes.
- [ ] **Build a conv-transpose autoencoder** — validation passes (proves "all types").
- [ ] Deliberate channel mismatch → validate names the right node/type; fix via `update_layer`; re-validate.
- [ ] Human edit in browser shows in agent's `get_architecture()`.
- [ ] `generate_code` output runs in a fresh `.py` outside the repo with only torch.
- [ ] Cycle attempt rejected; `reset_architecture` clears canvas, next id is `n1`.
- [ ] Demo records cleanly into the README GIF.

## Status
⬜ not started.
