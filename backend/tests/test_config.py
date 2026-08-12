"""backend/config.py: the one place env is parsed, read once."""

import dataclasses

import pytest

import config


def test_defaults():
    c = config.load({})
    assert c.mode == ""
    assert c.ws_port == 8765
    assert c.log_level == "INFO"
    assert c.log_file.endswith("nn_architect.log")  # repo logs dir
    assert c.validation_max_params == 500_000_000
    assert c.validation_mem_mb == 8192
    assert c.validation_timeout_s == 10.0


def test_standalone_mode_is_lowercased():
    c = config.load({"MODE": "Standalone"})
    assert c.mode == "standalone"


def test_ws_port_and_explicit_log_file_override():
    c = config.load({"WS_PORT": "9001", "LOG_FILE": "/tmp/x.log"})
    assert c.ws_port == 9001
    assert c.log_file == "/tmp/x.log"


def test_empty_log_file_disables_file_logging():
    c = config.load({"LOG_FILE": ""})
    assert c.log_file == ""


def test_validation_guards_are_env_tunable():
    c = config.load({"NN_VALIDATION_MAX_PARAMS": "1000", "NN_VALIDATION_MEM_MB": "512",
                     "NN_VALIDATION_TIMEOUT_S": "2.5"})
    assert c.validation_max_params == 1000
    assert c.validation_mem_mb == 512
    assert c.validation_timeout_s == 2.5


def test_config_is_frozen():
    c = config.load({})
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.ws_port = 1
