"""Deploy S1: /healthz liveness + CORS allowlist on the ws app."""

from fastapi.testclient import TestClient

from store import ArchitectureStore
from ws_app import build_app


def _client(cors_origins=()):
    app, _broadcast = build_app(ArchitectureStore(), cors_origins=cors_origins)
    return TestClient(app)


def test_healthz_ok():
    res = _client().get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_cors_headers_for_allowlisted_origin():
    client = _client(cors_origins=["https://app.example"])
    res = client.get("/healthz", headers={"Origin": "https://app.example"})
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "https://app.example"


def test_no_cors_headers_for_unlisted_origin():
    client = _client(cors_origins=["https://app.example"])
    res = client.get("/healthz", headers={"Origin": "https://evil.example"})
    # request still succeeds (CORS is a browser-side guard) but carries no
    # allow-origin header, so the browser blocks the read
    assert res.status_code == 200
    assert "access-control-allow-origin" not in res.headers


def test_no_origins_configured_means_no_cors_middleware():
    res = _client(cors_origins=()).get("/healthz", headers={"Origin": "https://app.example"})
    assert res.status_code == 200
    assert "access-control-allow-origin" not in res.headers
