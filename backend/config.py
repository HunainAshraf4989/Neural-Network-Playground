"""Central configuration: every env var the backend cares about, parsed once
into a frozen dataclass.

``load()`` is pure (env in, Config out) so tests can feed it a dict. Nothing
here reads the environment at import time; ``main.py`` calls ``load()`` once at
startup. Modules that must work standalone in a subprocess-free context
(``validator.py``, ``runner_template.py``) keep reading their own env vars —
the names below are the same ones, documented in one place.
"""

import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_FILE = str(_REPO_ROOT / "logs" / "nn_architect.log")


@dataclass(frozen=True)
class Config:
    # mode: "" = full (MCP stdio + websocket), "standalone" = websocket only
    mode: str
    ws_port: int
    log_level: str
    log_file: str  # "" disables file logging (stderr always on)

    # validation resource guards (validator.py / runner_template.py read the
    # same env vars themselves; these mirror them in one documented place)
    validation_max_params: int
    validation_mem_mb: int
    validation_timeout_s: float


def load(env=None) -> Config:
    """Parse ``env`` (default ``os.environ``) into a Config. Read once at
    process start; never re-read scattered through the codebase."""
    if env is None:
        env = os.environ
    return Config(
        mode=env.get("MODE", "").lower(),
        ws_port=int(env.get("WS_PORT") or "8765"),
        log_level=env.get("LOG_LEVEL", "INFO"),
        log_file=env.get("LOG_FILE", _DEFAULT_LOG_FILE),
        validation_max_params=int(env.get("NN_VALIDATION_MAX_PARAMS", str(500_000_000))),
        validation_mem_mb=int(env.get("NN_VALIDATION_MEM_MB", "8192")),
        validation_timeout_s=float(env.get("NN_VALIDATION_TIMEOUT_S", "10")),
    )
