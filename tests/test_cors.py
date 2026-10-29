"""CORS allows only the React demo's development origins (offline)."""

import pytest

ALLOWED = ["http://localhost:8443", "http://127.0.0.1:8443"]


def _preflight(client, origin, method="POST"):
    return client.options(
        "/api/chat",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": "content-type",
        },
    )


@pytest.mark.parametrize("origin", ALLOWED)
def test_frontend_origins_are_allowed(client, origin):
    resp = _preflight(client, origin)
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == origin
    assert "POST" in resp.headers["access-control-allow-methods"]
    assert "GET" in resp.headers["access-control-allow-methods"]


@pytest.mark.parametrize(
    "origin", ["http://evil.example", "http://localhost:3000", "https://localhost:8443"]
)
def test_other_origins_are_rejected(client, origin):
    resp = _preflight(client, origin)
    assert "access-control-allow-origin" not in resp.headers


def test_unneeded_method_and_wildcard_not_allowed(client):
    resp = _preflight(client, ALLOWED[0], method="DELETE")
    assert resp.status_code == 400
    simple = client.get("/health", headers={"Origin": "http://evil.example"})
    assert simple.headers.get("access-control-allow-origin") != "*"
    assert "access-control-allow-origin" not in simple.headers


def test_actual_response_carries_cors_header_for_allowed_origin(client):
    resp = client.get("/health", headers={"Origin": ALLOWED[0]})
    assert resp.headers["access-control-allow-origin"] == ALLOWED[0]
