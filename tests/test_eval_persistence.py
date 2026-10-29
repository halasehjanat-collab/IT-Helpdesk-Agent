import json
import sqlite3
from unittest.mock import MagicMock

import httpx

import app as app_module
from evals import run_evals
from evals import run_isolated_evals as iso
from services.local_ticket_store import SQLiteTicketMetadataStore, TicketStoreError
from tests.conftest import VALID_PAYLOAD


def _approve_seeded(client, seed_ticket_action):
    seed_ticket_action("s-a", "act-a")
    return client.post(
        "/api/approve",
        json={"session_id": "s-a", "action_id": "act-a", "approved": True},
    ).json()


# --- S2-05: the approved write must be reported as a create_ticket tool call
def test_approved_creation_reports_create_ticket_tool_call(client, fake_mcp, seed_ticket_action):
    body = _approve_seeded(client, seed_ticket_action)
    assert body["state"] == "ticket_created"
    assert body["tool_calls"] == [{"tool": "create_ticket", "ok": True}]
    assert fake_mcp.create_approved_ticket.await_count == 1


def test_rejection_and_chat_never_report_create_ticket(client, fake_mcp, seed_ticket_action):
    seed_ticket_action("s-r", "act-r")
    body = client.post(
        "/api/approve",
        json={"session_id": "s-r", "action_id": "act-r", "approved": False},
    ).json()
    assert body["tool_calls"] == []
    chat = client.post("/api/chat", json={"session_id": "s-r", "message": "hello"}).json()
    assert all(call.get("tool") != "create_ticket" for call in chat["tool_calls"])
    fake_mcp.create_approved_ticket.assert_not_called()


# --- S2-10: a recycled external id must be reported, not hidden
def test_duplicate_external_id_is_a_store_error_and_keeps_uniqueness(tmp_path):
    store = SQLiteTicketMetadataStore(tmp_path / "m.sqlite3")
    store.initialize()
    store.record_created_ticket(
        idempotency_key="k1", external_ticket_id="IT-2001", payload_sha256="a" * 64
    )
    try:
        store.record_created_ticket(
            idempotency_key="k2", external_ticket_id="IT-2001", payload_sha256="b" * 64
        )
    except TicketStoreError:
        pass
    else:
        raise AssertionError("a duplicate external id must not be stored")
    assert [r["idempotency_key"] for r in store.list_ticket_references()] == ["k1"]


def test_local_reference_failure_is_reported_and_not_a_clean_success(
    client, fake_mcp, seed_ticket_action, monkeypatch
):
    store = app_module.app.state.ticket_store
    monkeypatch.setattr(
        store,
        "record_created_ticket",
        MagicMock(side_effect=TicketStoreError("UNIQUE constraint failed")),
    )
    body = _approve_seeded(client, seed_ticket_action)
    assert body["state"] == "ticket_created_local_persistence_failed"
    assert "could not be saved" in body["reply"]
    assert body["result"] == {"ticket_id": "IT-2001"}
    assert fake_mcp.create_approved_ticket.await_count == 1


def test_eval_runner_fails_a_case_whose_local_reference_was_not_saved():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"reply": "Ticket IT-2001 was created, but",
                                              "state": "ticket_created_local_persistence_failed",
                                              "result": {"ticket_id": "IT-2001"}})
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    case = {"id": "X", "turns": [{"say": "hi"}], "expect": {"result_has": ["ticket_id"]}}
    result = run_evals.run_case(case, {"APP": "http://app"}, client, iso.CASES, None, False)
    assert result["passed"] is False
    assert any(c["check"] == "local_persistence" and c["ok"] is False for c in result["checks"])


# --- isolated runner: reference table reset and valid output paths
def test_clear_local_references_only_empties_the_temp_table(tmp_path):
    store = SQLiteTicketMetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()
    store.record_created_ticket(
        idempotency_key="k1", external_ticket_id="IT-2001", payload_sha256="a" * 64
    )
    iso.clear_local_references(tmp_path)
    assert store.list_ticket_references() == []
    # The id can now be recorded again, as after an eval reset.
    store.record_created_ticket(
        idempotency_key="k2", external_ticket_id="IT-2001", payload_sha256="b" * 64
    )
    with sqlite3.connect(tmp_path / "metadata.sqlite3") as connection:
        assert connection.execute("SELECT count(*) FROM ticket_references").fetchone()[0] == 1


def test_combined_scorecards_are_written_to_distinct_file_paths(tmp_path):
    results = [{"id": "S2-01", "category": "x", "passed": True, "latencies": [1.0],
                "cost_usd": 0.0, "error": None, "checks": [], "turns": []}]
    out = tmp_path / "isolated_scorecard.md"
    assert iso.write_combined(results, 0.0, out) == 0
    assert out.is_file() and out.with_suffix(".json").is_file()
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["summary"]["passed"] == 1
    assert iso.OUT_MD.name == "isolated_scorecard.md"
    assert iso.OUT_MD.parent == iso.ROOT / "evals"
    assert iso.OUT_MD != iso.ROOT / "evals" / "scorecard.md"


def test_case_ids_selects_requested_cases():
    assert iso.case_ids("S2-05,S2-10") == ["S2-05", "S2-10"]
    assert len(iso.case_ids(None)) == 10


# --- S2-05: KB-002 is a reference only when the assistant verifiably cited it
def _stub_cited_agent(monkeypatch, citations):
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        app_module,
        "run_agent",
        AsyncMock(
            return_value={
                "reply": "Run NakhlaManage Sync and restart [KB-002].",
                "citations": citations,
                "tool_calls": [],
                "pending_action": None,
            }
        ),
    )


def _s205_turns(client):
    user = {"email": "sara.mikhail@nakhla.example"}
    client.post("/api/chat", json={
        "session_id": "s205", "user": user,
        "message": "My NakhlaConnect VPN shows NX-112 since this morning."})
    return client.post("/api/chat", json={
        "session_id": "s205", "user": user,
        "message": "I did all of that: NakhlaManage Sync, restarted the laptop. "
                   "It still shows NX-112 and I cannot work. Please open a ticket."}).json()


def test_followed_cited_guidance_adds_the_cited_kb_reference(client, fake_mcp, monkeypatch):
    cite = [{"id": "KB-002", "title": "VPN", "filename": "KB-002.md"}]
    _stub_cited_agent(monkeypatch, cite)
    draft = _s205_turns(client)
    assert draft["state"] == "ticket_draft_pending"
    assert draft["pending_action"]["payload"]["kb_refs"] == ["KB-002"]
    fake_mcp.create_approved_ticket.assert_not_called()


def test_no_kb_reference_is_invented_without_a_prior_citation(client, fake_mcp, monkeypatch):
    _stub_cited_agent(monkeypatch, [])
    draft = _s205_turns(client)
    assert draft["pending_action"]["payload"]["kb_refs"] == []


def test_claim_of_following_guidance_does_not_cite_other_sessions(client, fake_mcp, monkeypatch):
    cite = [{"id": "KB-002", "title": "VPN", "filename": "KB-002.md"}]
    _stub_cited_agent(monkeypatch, cite)
    client.post("/api/chat", json={"session_id": "other", "message": "NX-112 on VPN"})
    _stub_cited_agent(monkeypatch, [])
    draft = _s205_turns(client)
    assert draft["pending_action"]["payload"]["kb_refs"] == []
