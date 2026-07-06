"""Central configuration: every env var the backend cares about, parsed once
into a frozen dataclass (deployment plan S0 — the first frozen contract).

``load()`` is pure (env in, Config out) so tests can feed it a dict. Nothing
here reads the environment at import time; ``main.py`` calls ``load()`` once at
startup. Modules that must work standalone in a subprocess-free context
(``validator.py``, ``runner_template.py``) keep reading their own env vars —
the names below are the same ones, documented in one place.

Fields for later stages (quotas, Supabase, CORS, demo doc) are parsed now so
the contract is complete; they are unused until S2/S3 wire them in.
"""

import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_FILE = str(_REPO_ROOT / "logs" / "nn_architect.log")


@dataclass(frozen=True)
class Config:
    # mode: "" = full (MCP stdio + websocket), "standalone" = websocket only,
    # "server" = deployed multi-user mode (S2+)
    mode: str
    ws_port: int
    log_level: str
    log_file: str  # "" disables file logging (stderr always on)

    # validation resource guards (validator.py / runner_template.py read the
    # same env vars themselves; these mirror them for server-mode consumers)
    validation_max_params: int
    validation_mem_mb: int
    validation_timeout_s: float

    # quotas (enforced from S2/S3 on)
    max_nodes_per_doc: int
    max_docs_per_user: int

    # auth / persistence (S2/S3; empty = disabled)
    supabase_url: str
    supabase_service_role_key: str
    supabase_jwt_secret: str

    cors_origins: tuple[str, ...]
    demo_doc_id: str


def load(env=None) -> Config:
    """Parse ``env`` (default ``os.environ``) into a Config. Read once at
    process start; never re-read scattered through the codebase."""
    if env is None:
        env = os.environ
    mode = env.get("MODE", "").lower()
    # containers run non-root and the repo dir may not be writable: server mode
    # defaults to stderr-only logging unless LOG_FILE is set explicitly
    default_log = "" if mode == "server" else _DEFAULT_LOG_FILE
    cors = tuple(o.strip() for o in env.get("CORS_ORIGINS", "").split(",") if o.strip())
    return Config(
        mode=mode,
        # WS_PORT wins locally; PORT is how hosts like HF Spaces pick the port
        ws_port=int(env.get("WS_PORT") or env.get("PORT") or "8765"),
        log_level=env.get("LOG_LEVEL", "INFO"),
        log_file=env.get("LOG_FILE", default_log),
        validation_max_params=int(env.get("NN_VALIDATION_MAX_PARAMS", str(500_000_000))),
        validation_mem_mb=int(env.get("NN_VALIDATION_MEM_MB", "4096")),
        validation_timeout_s=float(env.get("NN_VALIDATION_TIMEOUT_S", "10")),
        max_nodes_per_doc=int(env.get("NN_MAX_NODES_PER_DOC", "300")),
        max_docs_per_user=int(env.get("NN_MAX_DOCS_PER_USER", "20")),
        supabase_url=env.get("SUPABASE_URL", ""),
        supabase_service_role_key=env.get("SUPABASE_SERVICE_ROLE_KEY", ""),
        supabase_jwt_secret=env.get("SUPABASE_JWT_SECRET", ""),
        cors_origins=cors,
        demo_doc_id=env.get("DEMO_DOC_ID", ""),
    )
