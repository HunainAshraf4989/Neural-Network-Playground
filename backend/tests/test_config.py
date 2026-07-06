"""backend/config.py (deploy S0): the one place env is parsed, read once."""

import dataclasses

import pytest

import config


def test_defaults_match_pre_s0_behavior():
    c = config.load({})
    assert c.mode == ""
    assert c.ws_port == 8765
    assert c.log_level == "INFO"
    assert c.log_file.endswith("nn_architect.log")  # repo logs dir, as before
    assert c.validation_max_params == 500_000_000
    assert c.validation_mem_mb == 4096
    assert c.validation_timeout_s == 10.0
    assert c.cors_origins == ()


def test_server_mode_defaults_log_file_off_and_honors_port():
    c = config.load({"MODE": "server", "PORT": "7860"})
    assert c.mode == "server"
    assert c.ws_port == 7860  # HF Spaces-style PORT
    assert c.log_file == ""   # non-root container: stderr only unless LOG_FILE set


def test_ws_port_wins_over_port_and_explicit_log_file_sticks():
    c = config.load({"MODE": "server", "WS_PORT": "9001", "PORT": "7860",
                     "LOG_FILE": "/tmp/x.log"})
    assert c.ws_port == 9001
    assert c.log_file == "/tmp/x.log"


def test_cors_origins_parsed_from_csv():
    c = config.load({"CORS_ORIGINS": "https://a.app, https://b.app ,"})
    assert c.cors_origins == ("https://a.app", "https://b.app")


def test_config_is_frozen():
    c = config.load({})
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.ws_port = 1
