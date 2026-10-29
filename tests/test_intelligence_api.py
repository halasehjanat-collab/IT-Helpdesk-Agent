"""Offline tests for the token-gated, read-only /api/intelligence/* routes."""

from __future__ import annotations

import hashlib
import json
import socket
import sqlite3

import pytest
from fastapi.testclient import TestClient

import app as app_module
from services import ticket_intelligence as ti
from services.helpdesk_ticket_db import TicketDatabase

TOKEN = "unit-test-token-not-a-secret"
ROUTES = {
    "/api/intelligence/duplicates": {"subject": "VPN NX-781 again", "category": "network_access"},
    "/api/intelligence/trends": {},
    "/api/intelligence/knowledge-proposals": {},
}
SECRET_RESOLUTION = "SECRET-RESOLUTION-TEXT reset the user's home address record"
SECRET_DESCRIPTION = "SECRET-DESCRIPTION-TEXT with employee detail"
REQUESTER = "private.person@nakhla.example"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fields(subject, category="network_access", description=SECRET_DESCRIPTION):
    return {
        "subject": subject, "description": description, "category": category, "priority": "P3",
        "requester_email": REQUESTER, "department": "Sales", "affected_users": 1,
        "assigned_team": "Net", "attempted_steps": [], "kb_refs": ["KB-002"],
    }


@pytest.fixture
def intel_db(tmp_path, monkeypatch):
    """A temp ticket DB (never the default one) with synthetic tickets, plus a stub archive."""
    path = tmp_path / "intel_tickets.sqlite3"
    database = TicketDatabase(path)
    database.initialize()
    for index, subject in enumerate(["VPN NX-781 after reimage", "VPN NX-503 evening drops"]):
        database.create_ticket(_fields(subject), created_at=f"2026-10-0{index + 1}T08:00:00Z",
                               idempotency_key=None, request_hash=f"h{index}")
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE tickets SET status='resolved', resolution=?, resolved_at=created_at", (SECRET_RESOLUTION,)
        )
    archive = tmp_path / "archive.json"
    archive.write_text(json.dumps({"tickets": [{
        "ticket_id": "IT-1", "subject": "NX-781 client outdated", "description": "old " + SECRET_DESCRIPTION,
        "category": "network_access", "priority": "P3", "status": "resolved", "requester_email": REQUESTER,
        "department": "Sales", "affected_users": 1, "assigned_team": "Net", "attempted_steps": [],
        "kb_refs": ["KB-001", "KB-002"], "resolution": SECRET_RESOLUTION,
        "created_at": "2026-09-20T08:00:00Z", "resolved_at": "2026-09-20T09:00:00Z", "notes": [],
    }]}), encoding="utf-8")
    monkeypatch.setenv("HELPDESK_TICKET_DB_PATH", str(path))
    monkeypatch.setattr(ti, "DEFAULT_ARCHIVE_PATH", archive)
    return path


@pytest.fixture
def intel_client(intel_db, _guards, monkeypatch):
    monkeypatch.setenv(app_module.INTELLIGENCE_TOKEN_ENV, TOKEN)
    return TestClient(app_module.app)


AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def _guards(monkeypatch):
    real_connect = socket.socket.connect

    def refuse(*args, **kwargs):
        raise AssertionError("forbidden call from an intelligence route")

    def refuse_external(sock, address, *args, **kwargs):
        # The Windows event loop uses a loopback socket pair; anything else is external.
        if not (isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}):
            raise AssertionError(f"external network access attempted: {address!r}")
        return real_connect(sock, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", refuse_external)
    monkeypatch.setattr(app_module, "run_agent", refuse)
    monkeypatch.setattr(app_module.ReadOnlyMCPClient, "call_tool", refuse, raising=False)
    monkeypatch.setattr(TicketDatabase, "create_ticket", refuse)
    monkeypatch.setattr(TicketDatabase, "add_note", refuse)


# ------------------------------------------------------------------ disabled by default
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("token_value", [None, "", "   "])
def test_routes_are_disabled_without_a_token(route, token_value, _guards, monkeypatch):
    if token_value is None:
        monkeypatch.delenv(app_module.INTELLIGENCE_TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(app_module.INTELLIGENCE_TOKEN_ENV, token_value)
    client = TestClient(app_module.app)
    assert client.post(route, json=ROUTES[route], headers=AUTH).status_code == 404
    assert client.post(route, json={"bogus": 1}, headers=AUTH).status_code == 404  # no validation leak
    assert client.post(route, json=ROUTES[route]).status_code == 404


# ------------------------------------------------------------------ authentication
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("headers", [
    {}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}, {"Authorization": f"Basic {TOKEN}"},
    {"Authorization": "Bearer "}, {"Authorization": f"Bearer {TOKEN}x"}, {"X-Session-Id": "s1"},
])
def test_missing_or_wrong_token_is_rejected_without_echoing_secrets(route, headers, intel_client):
    response = intel_client.post(route, json=ROUTES[route], headers=headers)
    assert response.status_code == 401
    assert TOKEN not in response.text and response.json() == {"detail": "Authentication required."}
    assert response.headers["www-authenticate"] == "Bearer"


def test_session_id_or_email_in_body_is_not_authorization(intel_client):
    response = intel_client.post(
        "/api/intelligence/trends", json={"session_id": "s1", "user": {"email": REQUESTER}}
    )
    assert response.status_code == 401


def test_unauthenticated_invalid_body_gets_401_not_422(intel_client):
    assert intel_client.post("/api/intelligence/trends", json={"bogus": 1}).status_code == 401


def test_token_comparison_is_constant_time(monkeypatch, intel_client):
    calls = []
    real = app_module.hmac.compare_digest
    monkeypatch.setattr(app_module.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    intel_client.post("/api/intelligence/trends", json={}, headers=AUTH)
    assert calls


def test_token_is_never_logged(intel_client, caplog):
    caplog.set_level("DEBUG")
    intel_client.post("/api/intelligence/trends", json={}, headers={"Authorization": "Bearer wrong-secret-guess"})
    intel_client.post("/api/intelligence/trends", json={}, headers=AUTH)
    assert TOKEN not in caplog.text and "wrong-secret-guess" not in caplog.text


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("route,body", [
    ("/api/intelligence/duplicates", {"subject": "", "category": "x"}),
    ("/api/intelligence/duplicates", {"subject": "s", "category": "x", "similarity_threshold": 1.5}),
    ("/api/intelligence/duplicates", {"subject": "s", "category": "x", "unknown": 1}),
    ("/api/intelligence/duplicates", {"subject": "s" * 301, "category": "x"}),
    ("/api/intelligence/trends", {"period": "year"}),
    ("/api/intelligence/trends", {"group_by": "requester_email"}),
    ("/api/intelligence/trends", {"start": "yesterday"}),
    ("/api/intelligence/trends", {"extra": True}),
    ("/api/intelligence/knowledge-proposals", {"min_tickets": 0}),
    ("/api/intelligence/knowledge-proposals", {"approved": True}),
])
def test_invalid_requests_are_rejected(route, body, intel_client):
    assert intel_client.post(route, json=body, headers=AUTH).status_code == 422


def test_get_is_not_allowed(intel_client):
    assert intel_client.get("/api/intelligence/trends", headers=AUTH).status_code == 405


# ------------------------------------------------------------------ results and minimisation
def test_duplicates_return_evidence_without_text_or_personal_data(intel_client):
    response = intel_client.post("/api/intelligence/duplicates", headers=AUTH, json={
        "subject": "NX-781 on login", "category": "network_access", "requester_email": REQUESTER,
        "created_at": "2026-10-05T08:00:00Z"})
    assert response.status_code == 200
    body = response.json()
    assert body["automatic_action"] == "none" and "note" in body
    assert body["source"]["ticket_counts"] == {"historical_archive": 1, "ticket_database": 2}
    ids = {c["ticket_id"] for c in body["candidates"]}
    assert "IT-1" in ids and "IT-2002" not in ids  # NX-503 ticket conflicts on error code
    assert all(c["reasons"] and c["evidence"]["shared_error_codes"] == ["NX-781"] for c in body["candidates"])
    for leaked in (SECRET_DESCRIPTION, SECRET_RESOLUTION, REQUESTER):
        assert leaked not in response.text


def test_trends_report_limitations_and_no_predictions(intel_client):
    body = intel_client.post("/api/intelligence/trends", headers=AUTH, json={}).json()
    assert body["sample_size"] == 3 and body["insufficient_data"] is True
    assert body["limitations"] and body["predictions"].startswith("none")
    assert body["statistical_significance"] == "not assessed" and body["observed_changes"] == []
    assert body["source"]["read_only"] is True and body["time_range"]["first_ticket"]
    for leaked in (SECRET_DESCRIPTION, SECRET_RESOLUTION, REQUESTER):
        assert leaked not in json.dumps(body)


def test_proposals_withhold_resolution_text_and_are_never_publishable(intel_client):
    response = intel_client.post("/api/intelligence/knowledge-proposals", headers=AUTH, json={})
    body = response.json()
    assert response.status_code == 200 and body["automatic_action"].startswith("none")
    assert body["proposals_returned"] == len(body["proposals"]) <= body["proposals_total"]
    nx781 = next(p for p in body["proposals"] if p["error_code"] == "NX-781")
    assert nx781["verified"] is False and nx781["publishable"] is False and nx781["requires_human_review"] is True
    assert nx781["resolution_text_included"] is False
    assert nx781["source_ticket_ids"] == ["IT-1", "IT-2001"]
    assert all("resolution_excerpt" not in item for item in nx781["evidence"])
    assert {r["id"] for r in nx781["kb_references"]["ineligible"]} == {"KB-001"}  # preserved uncertainty
    assert nx781["uncertainty"]
    for leaked in (SECRET_DESCRIPTION, SECRET_RESOLUTION, REQUESTER):
        assert leaked not in response.text


def test_max_proposals_caps_output(intel_client):
    body = intel_client.post("/api/intelligence/knowledge-proposals", headers=AUTH, json={"max_proposals": 1}).json()
    assert body["proposals_returned"] == 1 and body["proposals_total"] >= 2


# ------------------------------------------------------------------ error handling
def test_corrupt_database_gives_generic_503(intel_client, intel_db):
    intel_db.write_bytes(b"not a sqlite database" * 20)
    response = intel_client.post("/api/intelligence/trends", headers=AUTH, json={})
    assert response.status_code == 503
    assert response.json() == {"detail": "Ticket data is currently unavailable."}
    assert str(intel_db) not in response.text and TOKEN not in response.text


def test_missing_database_is_not_created_and_archive_still_answers(intel_client, intel_db, monkeypatch, tmp_path):
    missing = tmp_path / "nothing-here.sqlite3"
    monkeypatch.setenv("HELPDESK_TICKET_DB_PATH", str(missing))
    body = intel_client.post("/api/intelligence/trends", headers=AUTH, json={}).json()
    assert body["source"]["ticket_counts"] == {"historical_archive": 1, "ticket_database": 0}
    assert not missing.exists()


def test_unreadable_manifest_gives_generic_503(intel_client, monkeypatch):
    def boom(*args, **kwargs):
        raise ti.IntelligenceDataError("secret path detail")

    monkeypatch.setattr(ti.KnowledgeCatalog, "load", classmethod(boom))
    response = intel_client.post("/api/intelligence/knowledge-proposals", headers=AUTH, json={})
    assert response.status_code == 503 and "secret path detail" not in response.text


# ------------------------------------------------------------------ read-only and isolation
def test_routes_leave_databases_unchanged_and_call_no_writers(intel_client, intel_db):
    before = _sha(intel_db)
    for route, body in ROUTES.items():
        assert intel_client.post(route, json=body, headers=AUTH).status_code == 200
    assert _sha(intel_db) == before  # create_ticket, add_note, run_agent and sockets are also blocked by fixture


def test_default_project_databases_are_not_touched(intel_client):
    metadata = app_module.Path(app_module.__file__).parent / "services" / "data" / "helpdesk_metadata.sqlite3"
    before = _sha(metadata) if metadata.exists() else None
    default_tickets = app_module.DEFAULT_TICKET_DB_PATH
    tickets_before = _sha(default_tickets) if default_tickets.exists() else None
    for route, body in ROUTES.items():
        intel_client.post(route, json=body, headers=AUTH)
    assert (_sha(metadata) if metadata.exists() else None) == before
    # A legitimate existing ticket database is allowed; it must be unchanged.
    assert (_sha(default_tickets) if default_tickets.exists() else None) == tickets_before


def test_intelligence_is_not_reachable_from_chat_or_approve(client, fake_mcp, _guards, monkeypatch):
    calls = []
    for name in ("find_duplicate_candidates", "summarize_trends", "propose_knowledge_updates", "load_ticket_dataset"):
        monkeypatch.setattr(ti, name, lambda *a, _n=name, **k: calls.append(_n))

    async def fake_agent(message, history=None):
        return {"answer": "ok", "reply": "ok", "citations": [], "tool_calls": [], "pending_action": None,
                "usage": None, "state": "completed"}

    monkeypatch.setattr(app_module, "run_agent", fake_agent)
    chat = client.post("/api/chat", json={"session_id": "s1", "message": "Show duplicate tickets and trends"})
    assert chat.status_code == 200 and set(chat.json()) >= {"reply", "pending_action", "citations", "tool_calls"}
    approve = client.post("/api/approve", json={"session_id": "s1", "action_id": "none", "approved": True})
    assert approve.json()["state"] == "no_pending_action"
    fake_mcp.create_approved_ticket.assert_not_called()
    assert calls == []
    assert client.get("/health").json() == {"status": "ok"}


def test_agent_module_does_not_reference_intelligence():
    agent_source = (app_module.Path(app_module.__file__).parent / "agent.py").read_text(encoding="utf-8")
    assert "ticket_intelligence" not in agent_source
