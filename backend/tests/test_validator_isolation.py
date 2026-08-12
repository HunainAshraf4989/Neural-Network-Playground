"""validator.py process hygiene: per-invocation temp files and the
NN_VALIDATION_TIMEOUT_S override. The runner subprocess is faked here — real
execution (and hence shape correctness) is covered by the validate tests in
test_store.py; this file only cares that concurrent validations never share a
generated-code path and that the file is gone afterwards, success or timeout.
"""

import importlib
import os
import subprocess
import threading
from types import SimpleNamespace

import validator


def _tiny_state():
    return {
        "nodes": [
            {"id": "n1", "type": "input", "params": {"shape": [1, 4], "dtype": "float32"}},
            {"id": "n2", "type": "relu", "params": {}},
        ],
        "edges": [{"from": "n1", "to": "n2"}],
    }


def test_concurrent_validations_get_distinct_temp_files(monkeypatch):
    barrier = threading.Barrier(2, timeout=10)
    seen = []

    def fake_run(cmd, **kwargs):
        path = cmd[2]  # [python, runner, generated_path, spec]
        barrier.wait()  # both invocations in flight at once
        assert os.path.exists(path), "generated file must exist while running"
        seen.append(path)
        barrier.wait()
        return SimpleNamespace(stdout='{"ok": true, "output_shapes": [[2, 4]]}', stderr="")

    monkeypatch.setattr(validator.subprocess, "run", fake_run)

    results = []
    threads = [threading.Thread(target=lambda: results.append(validator.validate(_tiny_state())))
               for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seen) == 2 and seen[0] != seen[1], "temp paths must be per-invocation"
    assert all(r["valid"] for r in results)
    for path in seen:
        assert not os.path.exists(path), "generated file must be deleted afterwards"


def test_temp_file_deleted_even_on_timeout(monkeypatch):
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd[2])
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(validator.subprocess, "run", fake_run)
    res = validator.validate(_tiny_state())
    assert res["valid"] is False
    assert "timed out" in res["error"]["message"]
    assert len(seen) == 1 and not os.path.exists(seen[0])


def test_timeout_is_env_tunable(monkeypatch):
    monkeypatch.setenv("NN_VALIDATION_TIMEOUT_S", "3.5")
    try:
        importlib.reload(validator)
        assert validator.TIMEOUT_SECONDS == 3.5
    finally:
        monkeypatch.undo()
        importlib.reload(validator)
    assert validator.TIMEOUT_SECONDS == 10.0
