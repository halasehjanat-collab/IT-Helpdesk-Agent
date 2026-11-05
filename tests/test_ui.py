"""Offline checks that ui/app_ui.py matches the real /api/chat and /api/approve contracts.

The UI's httpx.post is routed into the FastAPI app through TestClient; no network, no real
tickets (the MCP writer is an AsyncMock from conftest).
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

import app as app_module
from tests.conftest import ticket_result

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
import app_ui


@pytest.fixture
def ui(client, monkeypatch):
    def post(url, json=None, timeout=None):
        path = url.removeprefix(app_ui.APP_URL)
        response = client.post(path, json=json)
        return httpx.Response(response.status_code, json=response.json())

    monkeypatch.setattr(app_ui.httpx, "post", post)
    return app_ui


def test_chat_sends_session_and_email_and_shows_reply(ui, client, monkeypatch):
    agent = AsyncMock(
        return_value={
            "reply": "Restart the VPN client [KB-002].",
            "citations": [{"id": "KB-002", "title": "VPN"}],
            "tool_calls": [],
            "pending_action": None,
        }
    )
    monkeypatch.setattr(app_module, "run_agent", agent)

    out = ui.send("my vpn fails", [], "ui-abc", " Dana.Levi@Nakhla.example ")
    history, pending, card, _, _, cites, tools, _ = out[1:]

    assert history[-1] == {"role": "assistant", "content": "Restart the VPN client [KB-002]."}
    assert pending is None and card == ui.NO_CARD
    assert "KB-002" in cites and "none" in tools


def test_tool_calls_without_timing_do_not_show_question_marks(ui):
    _, tools, _ = ui.side_panels({"tool_calls": [{"tool": "get_service_status", "ok": True}]})
    assert "get_service_status" in tools and "ok" in tools
    assert "? ms" not in tools and " ms" not in tools


def test_approve_sends_contract_fields_and_shows_real_ticket_id(ui, fake_mcp, seed_ticket_action):
    fake_mcp.create_approved_ticket = AsyncMock(return_value=ticket_result("IT-7777"))
    action = seed_ticket_action("ui-s1", "act-ui")

    out = ui.decide(True, [], "ui-s1", action)
    history, pending, card, _, _, _, _, result = out

    assert "IT-7777" in history[-1]["content"] and "IT-7777" in result
    assert pending is None and card == ui.NO_CARD
    assert fake_mcp.create_approved_ticket.await_count == 1


def test_repeated_approve_click_never_creates_a_second_ticket(ui, fake_mcp, seed_ticket_action):
    fake_mcp.create_approved_ticket = AsyncMock(return_value=ticket_result("IT-7777"))
    action = seed_ticket_action("ui-s1", "act-ui")

    ui.decide(True, [], "ui-s1", action)
    second = ui.decide(True, [], "ui-s1", action)

    assert fake_mcp.create_approved_ticket.await_count == 1
    assert "was created" not in second[0][-1]["content"] or "already" in second[0][-1]["content"].lower()


def test_reject_never_creates_a_ticket(ui, fake_mcp, seed_ticket_action):
    action = seed_ticket_action("ui-s1", "act-ui")
    out = ui.decide(False, [], "ui-s1", action)
    assert "was created" not in out[0][-1]["content"]
    fake_mcp.create_approved_ticket.assert_not_called()


def test_failed_creation_keeps_draft_for_retry_and_claims_no_ticket(ui, fake_mcp, seed_ticket_action):
    fake_mcp.create_approved_ticket = AsyncMock(side_effect=RuntimeError("HTTP 503"))
    action = seed_ticket_action("ui-s1", "act-ui")

    out = ui.decide(True, [], "ui-s1", action)

    assert "was created" not in out[0][-1]["content"].replace("not confirmed", "")
    assert out[1] == action  # draft kept so the user can retry the same action
    assert out[3]["interactive"] is True


def test_connection_error_is_reported_and_keeps_the_draft(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(app_ui.httpx, "post", boom)
    action = {"action_id": "a1", "summary": "s", "payload": {}}

    out = app_ui.decide(True, [], "ui-s1", action)

    assert "Cannot reach" in out[0][-1]["content"]
    assert out[1] == action
    sent = app_ui.send("hello", [], "ui-s1", "", action)
    assert "Cannot reach" in sent[1][-1]["content"] and sent[2] == action


def test_http_error_is_reported_without_claiming_success(monkeypatch):
    monkeypatch.setattr(
        app_ui.httpx, "post", lambda *a, **k: httpx.Response(500, json={"detail": "boom"})
    )
    out = app_ui.send("hello", [], "ui-s1", "")
    assert "HTTP 500" in out[1][-1]["content"] and out[2] is None


def test_approve_without_draft_does_not_call_backend(monkeypatch):
    monkeypatch.setattr(app_ui.httpx, "post", lambda *a, **k: pytest.fail("called backend"))
    out = app_ui.decide(True, [], "ui-s1", None)
    assert "no draft" in out[0][-1]["content"].lower()


def test_sign_in_shows_name_and_starts_a_new_session(ui, monkeypatch):
    monkeypatch.setattr(
        ui, "call_app",
        lambda path, body: {"verified": True, "email": body["email"], "name": "Tariq Hamdan",
                            "title": "Analyst", "department": "Finance"},
    )

    profile, signed, session_id, chat, pending, card, *_ = ui.sign_in(" Tariq.Hamdan@Nakhla.example ")

    assert "Signed in as Tariq Hamdan" in profile and "Finance" in profile
    assert signed == "tariq.hamdan@nakhla.example"
    assert session_id.startswith("ui-") and "Tariq Hamdan" in chat[0]["content"]
    assert pending is None and card == ui.NO_CARD


def test_sign_in_with_unknown_email_stays_anonymous(ui, monkeypatch):
    monkeypatch.setattr(ui, "call_app", lambda path, body: {"verified": False, "message": "Not in directory."})

    profile, signed, _, chat, *_ = ui.sign_in("nobody@nakhla.example")

    assert "Not signed in" in profile and "Not in directory." in profile
    assert signed == "" and chat == []


def test_sign_in_through_the_real_api(ui):
    profile, signed, *_ = ui.sign_in("dana.levi@nakhla.example")

    assert signed == "dana.levi@nakhla.example" and "Signed in as" in profile


def test_empty_email_does_not_call_backend(ui, monkeypatch):
    monkeypatch.setattr(ui, "call_app", lambda *a: pytest.fail("backend must not be called"))

    profile, signed, *_ = ui.sign_in("  ")

    assert signed == "" and "Enter your work email" in profile


def test_new_conversation_changes_session_and_clears_chat(ui):
    session_id, chat, pending, card, *_ = ui.new_conversation("tariq.hamdan@nakhla.example")

    assert session_id.startswith("ui-") and chat == [] and pending is None and card == ui.NO_CARD


def test_default_email_is_not_hardcoded(ui):
    assert "tariq.hamdan" not in Path(ui.__file__).read_text(encoding="utf-8").split("with gr.Blocks")[1].split("placeholder")[0]
