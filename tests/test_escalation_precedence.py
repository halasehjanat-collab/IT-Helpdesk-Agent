"""Escalation confirmation must beat a pending troubleshooting question (offline, mocked)."""

from unittest.mock import AsyncMock

import pytest

import app as app_module
from services.chat_memory import ChatMemoryStore

VPN_REPORT = "I cannot connect to the VPN. I restarted my laptop, but the problem persists."
TROUBLESHOOTING_QUESTION = (
    "Which error code does NakhlaConnect show, and which department are you in?"
)
SESSION = "s-escalate"


@pytest.fixture
def store(tmp_path):
    s = ChatMemoryStore(tmp_path / "chat_memory.sqlite3")
    s.initialize()
    return s


@pytest.fixture
def memory_app(client, fake_mcp, store):
    previous = getattr(app_module.app.state, "memory_store", None)
    app_module.app.state.memory_store = store
    yield client
    app_module.app.state.memory_store = previous


@pytest.fixture
def agent_mock(monkeypatch):
    # The model always re-asks its troubleshooting question if it is consulted.
    mock = AsyncMock(
        return_value={
            "reply": TROUBLESHOOTING_QUESTION,
            "citations": [],
            "tool_calls": [],
            "pending_action": None,
        }
    )
    monkeypatch.setattr(app_module, "run_agent", mock)
    return mock


def _chat(client, message, email=None):
    body = {"session_id": SESSION, "message": message}
    if email:
        body["user"] = {"email": email}
    return client.post("/api/chat", json=body).json()


def _seed_vpn_question(store):
    store.add_turn(SESSION, VPN_REPORT, TROUBLESHOOTING_QUESTION)


def _intake_text():
    intake = app_module.app.state.ticket_intakes.get(SESSION)
    return "\n".join(intake.messages) if intake else ""


def test_explicit_escalation_after_troubleshooting_question(
    memory_app, store, agent_mock, fake_mcp
):
    _seed_vpn_question(store)

    body = _chat(memory_app, "Yes, help me escalate this issue.")

    assert body["state"] == "ticket_info_required"
    assert body["pending_action"] is None
    assert "error code" not in body["reply"].lower()
    assert "department are you" not in body["reply"].lower()
    agent_mock.assert_not_awaited()
    fake_mcp.create_approved_ticket.assert_not_called()


def test_vpn_issue_is_preserved_as_ticket_topic(memory_app, store, agent_mock):
    _seed_vpn_question(store)

    _chat(memory_app, "Yes, help me escalate this issue.")

    assert "cannot connect to the VPN" in _intake_text()


def test_only_one_work_email_prompt(memory_app, store, agent_mock):
    _seed_vpn_question(store)

    body = _chat(memory_app, "Yes, help me escalate this issue.")

    assert body["reply"].lower().count("work email") == 1
    # One step at a time: nothing else is requested before the requester is verified.
    assert body["reply"].count("\n- ") == 1


def test_email_turn_completes_draft_with_vpn_topic_and_no_repeat(
    memory_app, store, agent_mock, fake_mcp
):
    _seed_vpn_question(store)
    _chat(memory_app, "Yes, help me escalate this issue.")

    body = _chat(memory_app, "dana.levi@nakhla.example")

    assert body["state"] == "ticket_draft_pending"
    assert "error code" not in body["reply"].lower()
    assert "work email" not in body["reply"].lower()
    payload = body["pending_action"]["payload"]
    assert payload["requester_email"] == "dana.levi@nakhla.example"
    assert "cannot connect to the VPN" in payload["description"]
    agent_mock.assert_not_awaited()
    # A draft only: nothing is written until /api/approve.
    fake_mcp.create_approved_ticket.assert_not_called()


def test_yes_after_explicit_escalation_offer_starts_intake(
    memory_app, store, agent_mock
):
    store.add_turn(
        SESSION, VPN_REPORT, "I can help you escalate this issue to the Service Desk."
    )

    body = _chat(memory_app, "yes")

    assert body["state"] == "ticket_info_required"
    assert "cannot connect to the VPN" in _intake_text()
    agent_mock.assert_not_awaited()


def test_yes_without_escalation_offer_does_not_start_intake(
    memory_app, store, agent_mock
):
    _seed_vpn_question(store)

    body = _chat(memory_app, "yes")

    assert "state" not in body
    assert SESSION not in app_module.app.state.ticket_intakes
    agent_mock.assert_awaited_once()


def test_invalid_work_email_is_still_rejected(memory_app, store, agent_mock, fake_mcp):
    _seed_vpn_question(store)

    body = _chat(
        memory_app, "Yes, help me escalate this issue.", email="someone@gmail.com"
    )

    assert body["state"] == "requester_unverified"
    assert body["reply"].count("not a Nakhla work address") == 1
    assert body["reply"].count("Nakhla work email") == 1
    assert body["pending_action"] is None
    fake_mcp.create_approved_ticket.assert_not_called()


def test_exact_conversation_from_first_message_through_pending_draft(
    memory_app, store, agent_mock, fake_mcp
):
    # Turn 1 goes through the real chat path; the model is mocked to ask its question.
    first = _chat(memory_app, VPN_REPORT)
    assert "state" not in first
    assert agent_mock.await_count == 1

    # Turn 2: explicit escalation must not consult the model again.
    second = _chat(memory_app, "Yes, help me escalate this issue.")
    assert agent_mock.await_count == 1
    assert second["state"] == "ticket_info_required"
    assert "error code" not in second["reply"].lower()
    assert second["reply"].lower().count("work email") == 1
    assert "cannot connect to the VPN" in _intake_text()

    # Turn 3: a non-Nakhla address is valid text but not a verifiable employee.
    bad = _chat(memory_app, "someone@gmail.com", email="someone@gmail.com")
    assert bad["state"] == "requester_unverified"
    assert bad["pending_action"] is None
    assert bad["reply"].count("Nakhla work email") == 1

    # Turn 4: a directory-verified address yields a draft about the VPN, unsubmitted.
    good = _chat(memory_app, "dana.levi@nakhla.example")
    assert good["state"] == "ticket_draft_pending"
    assert "VPN" in good["pending_action"]["payload"]["description"]
    assert agent_mock.await_count == 1
    fake_mcp.create_approved_ticket.assert_not_called()


@pytest.mark.parametrize(
    "message",
    [
        "Please prepare a ticket for my VPN problem",
        "I need a ticket for my VPN issue",
        "Please escalate my VPN issue",
    ],
)
def test_explicit_ticket_request_enters_intake_without_calling_the_model(
    memory_app, store, agent_mock, message
):
    body = _chat(memory_app, message)

    assert body["state"] == "ticket_info_required"
    assert "VPN" in _intake_text()
    assert body["reply"].lower().count("work email") == 1
    assert "error code" not in body["reply"].lower()
    agent_mock.assert_not_awaited()


def test_generic_support_ticket_request_asks_for_the_issue_without_the_model(
    memory_app, store, agent_mock
):
    body = _chat(memory_app, "I need to create a support ticket")

    assert body["state"] == "ticket_info_required"
    assert body["pending_action"] is None
    agent_mock.assert_not_awaited()


def test_new_ticket_request_does_not_inherit_unrelated_previous_issue(
    memory_app, store, agent_mock
):
    _seed_vpn_question(store)

    _chat(memory_app, "Please create a ticket for my freezing laptop")

    assert "freezing laptop" in _intake_text()
    assert "VPN" not in _intake_text()


@pytest.mark.parametrize("reply", ["Please proceed", "Yes, please proceed.", "Go ahead"])
def test_equivalent_confirmations_continue_intake_after_an_offer(
    memory_app, store, agent_mock, reply
):
    store.add_turn(
        SESSION, VPN_REPORT, "I can help you escalate this issue to the Service Desk."
    )

    body = _chat(memory_app, reply)

    assert body["state"] == "ticket_info_required"
    assert "cannot connect to the VPN" in _intake_text()
    agent_mock.assert_not_awaited()


def test_accepted_email_is_not_requested_again(memory_app, store, agent_mock):
    _seed_vpn_question(store)
    _chat(memory_app, "Yes, help me escalate this issue.")

    body = _chat(memory_app, "dana.levi@nakhla.example")

    assert "work email" not in body["reply"].lower()
    assert app_module.app.state.ticket_intakes.get(SESSION) is None or (
        app_module.app.state.ticket_intakes[SESSION].requester_email
        == "dana.levi@nakhla.example"
    )
