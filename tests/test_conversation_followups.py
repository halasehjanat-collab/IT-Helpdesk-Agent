"""Regression tests for follow-up handling in the live conversation (offline, mocked)."""

from unittest.mock import AsyncMock

import pytest

import app as app_module
from services import answer_verifier as av
from services.chat_memory import ChatMemoryStore

INTAKE_STATES = {
    "ticket_info_required",
    "requester_unverified",
    "ticket_draft_pending",
}


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
    mock = AsyncMock(
        return_value={
            "reply": "I could not confirm a KB solution.",
            "citations": [],
            "tool_calls": [],
            "pending_action": None,
        }
    )
    monkeypatch.setattr(app_module, "run_agent", mock)
    return mock


def _chat(client, message, session="s-follow", email=None):
    body = {"session_id": session, "message": message}
    if email:
        body["user"] = {"email": email}
    return client.post("/api/chat", json=body).json()


def _sent_message(agent_mock):
    return agent_mock.await_args.kwargs["message"]


def _intake_text(session="s-follow"):
    intake = app_module.app.state.ticket_intakes.get(session)
    return "\n".join(intake.messages) if intake else ""


# 1/2/5: a short confirmation continues the escalation workflow


def test_yes_after_escalation_offer_starts_ticket_intake(
    memory_app, store, agent_mock, fake_mcp
):
    store.add_turn("s-follow", "My VPN keeps failing", av.ABSTENTION)

    body = _chat(memory_app, "yes help me")

    assert body["state"] in INTAKE_STATES
    assert body["pending_action"] is None
    assert "My VPN keeps failing" in _intake_text("s-follow")
    agent_mock.assert_not_awaited()
    fake_mcp.create_approved_ticket.assert_not_called()
    assert "ticket has been created" not in body["reply"].lower().replace(
        "no ticket has been created", ""
    )


def test_ticket_intake_asks_for_work_email_when_missing(
    memory_app, store, agent_mock
):
    store.add_turn("s-follow", "My laptop keeps freezing", av.PARTIAL_NOTE)

    body = _chat(memory_app, "yes")

    assert body["state"] == "ticket_info_required"
    assert "work email" in body["reply"]


def test_invalid_email_is_still_rejected_during_intake(
    memory_app, store, agent_mock, fake_mcp
):
    store.add_turn("s-follow", "My laptop keeps freezing", av.PARTIAL_NOTE)

    body = _chat(memory_app, "yes", email="someone@gmail.com")

    assert body["state"] == "requester_unverified"
    assert body["pending_action"] is None
    assert "not a Nakhla work address" in body["reply"]
    fake_mcp.create_approved_ticket.assert_not_called()


def test_yes_without_a_pending_offer_is_ordinary_chat(
    memory_app, store, agent_mock
):
    store.add_turn("s-follow", "How do I reset VPN?", "Use Reset Profile [KB-002].")

    body = _chat(memory_app, "yes")

    assert "state" not in body
    assert _sent_message(agent_mock) == "yes"


def test_yes_with_no_history_is_ordinary_chat(client, fake_mcp, agent_mock):
    body = _chat(client, "yes")

    assert "state" not in body
    assert body["pending_action"] is None


def test_intake_turn_is_remembered_so_a_later_yes_is_not_reused(
    memory_app, store, agent_mock
):
    store.add_turn("s-follow", "My VPN keeps failing", av.ABSTENTION)
    _chat(memory_app, "yes help me")
    history = store.history("s-follow")

    assert history[-1]["role"] == "assistant"
    assert "work email" in history[-1]["content"]
    assert store.relevant_memories("s-follow", "VPN keeps failing") == []


# 4: a new issue must not inherit unrelated earlier context


def test_new_issue_ticket_request_ignores_unrelated_previous_topic(
    memory_app, store, agent_mock
):
    store.add_turn(
        "s-follow", "How do I reset my password?", "Use self-service [KB-001]."
    )

    _chat(memory_app, "Please create a ticket for my freezing laptop")

    intake = _intake_text("s-follow")
    assert "freezing laptop" in intake
    assert "password" not in intake.lower()
    agent_mock.assert_not_awaited()


def test_bare_ticket_request_still_uses_previous_issue(
    memory_app, store, agent_mock
):
    store.add_turn("s-follow", "My VPN keeps failing", av.ABSTENTION)

    _chat(memory_app, "Please prepare it for me")

    assert "My VPN keeps failing" in _intake_text("s-follow")


@pytest.mark.parametrize(
    "message,bare",
    [
        ("Please create a ticket for my freezing laptop", False),
        ("Create a ticket", True),
        ("Please prepare it for me.", True),
        ("I did all of that, please create a ticket", True),
    ],
)
def test_bare_ticket_request_detection(message, bare):
    assert app_module._is_bare_ticket_request(message) is bare


# 5: ticket intents that used to fall through to "I cannot create tickets"


@pytest.mark.parametrize(
    "message",
    [
        "I need a ticket for my laptop",
        "I want to open a support ticket",
        "Please escalate this to the service desk",
    ],
)
def test_more_ticket_phrasings_enter_the_draft_workflow(message):
    assert app_module._requests_ticket_creation(message)


@pytest.mark.parametrize(
    "message",
    [
        "How do I escalate a ticket?",
        "I don't want a ticket, just help me",
        'The text says "please escalate this".',
    ],
)
def test_questions_and_quotes_do_not_enter_the_draft_workflow(message):
    assert not app_module._requests_ticket_creation(message)


def test_system_prompt_does_not_tell_model_drafting_is_unavailable():
    import inspect

    import agent

    source = inspect.getsource(agent)
    assert "never say that ticket drafting is unavailable" in source
    assert "Never claim that a ticket was created" in source


# 3: numbering


def test_numbered_steps_are_renumbered_after_a_step_is_removed():
    lines = ["Steps:", "1. Open the client", "3. Sign in", "4. Reconnect"]
    assert av._renumber(lines) == [
        "Steps:",
        "1. Open the client",
        "2. Sign in",
        "3. Reconnect",
    ]


def test_renumbering_keeps_separate_lists_and_start_numbers():
    lines = ["2. B", "3. C", "Note", "1. X", "3. Z", "- bullet"]
    assert av._renumber(lines) == ["2. B", "3. C", "Note", "1. X", "2. Z", "- bullet"]


def test_verified_reply_with_removed_step_keeps_consecutive_numbers():
    from tests.test_answer_verifier import doc

    reply = (
        "1. Open the Nakhla VPN client and choose Reset Profile [KB-002].\n"
        "2. Reinstall the printer firmware from the vendor portal [KB-002].\n"
        "3. Sign in again with your employee account [KB-002]."
    )
    out = av.verify_answer(reply, [doc()], message="vpn")

    assert out.status == "partial"
    steps = [l for l in out.reply.splitlines() if l[:2] in ("1.", "2.", "3.")]
    assert [l[:2] for l in steps] == ["1.", "2."]
    assert "printer firmware" not in "\n".join(steps)
    assert av.UNVERIFIED_HEADER in out.reply
    assert "printer firmware" in out.reply.split(av.UNVERIFIED_HEADER)[1]
    assert out.reply.count(av.UNVERIFIED_HEADER) == 1
    assert out.reply.count(av.PARTIAL_NOTE) == 1


def test_real_ticket_creation_still_only_happens_through_approve(
    memory_app, store, agent_mock, fake_mcp
):
    store.add_turn("s-follow", "My VPN keeps failing", av.ABSTENTION)
    _chat(memory_app, "yes help me")
    _chat(memory_app, "yes")

    fake_mcp.create_approved_ticket.assert_not_called()
