import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from mcp.types import CallToolResult, TextContent

import app as app_module
from tests.conftest import VALID_PAYLOAD, ticket_result
from services.local_ticket_store import (
    TicketStoreError,
    payload_digest,
)

CHAT_FIELDS = {"reply", "pending_action", "citations", "tool_calls", "result"}


def approve(client, session_id, action_id, approved):
    return client.post(
        "/api/approve",
        json={"session_id": session_id, "action_id": action_id, "approved": approved},
    )


def draft(client, session_id="s1"):
    r = client.post("/api/chat", json={"session_id": session_id, "message": "/test-draft"})
    assert r.status_code == 200
    return r.json()["pending_action"]["action_id"]


# --- health and chat ---------------------------------------------------------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_normal_chat_contract(client, fake_mcp):
    r = client.post("/api/chat", json={"session_id": "s1", "message": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == CHAT_FIELDS
    assert body["pending_action"] is None
    assert body["citations"] == []
    assert body["tool_calls"] == []
    assert body["result"] == {}
    assert isinstance(body["reply"], str) and body["reply"]
    fake_mcp.create_approved_ticket.assert_not_called()
    assert app_module.app.state.ticket_store.list_ticket_references() == []


@pytest.mark.parametrize("body", [{"message": "hi"}, {"session_id": "s1"}])
def test_chat_requires_fields(client, body):
    assert client.post("/api/chat", json=body).status_code == 422


# --- test-only draft flow ----------------------------------------------------


def test_test_draft_contract(client, fake_mcp):
    r = client.post("/api/chat", json={"session_id": "s1", "message": "/test-draft"})
    body = r.json()
    assert set(body) == CHAT_FIELDS | {"state"}
    assert body["state"] == "test_draft_pending"
    action = body["pending_action"]
    assert action["test_only"] is True
    assert action["action_id"].startswith("test-")
    fake_mcp.create_approved_ticket.assert_not_called()


def test_test_draft_approve_never_writes(client, fake_mcp):
    action_id = draft(client)
    body = approve(client, "s1", action_id, True).json()
    assert body["state"] == "approved"
    assert body["result"] == {}
    assert body["pending_action"] is None
    fake_mcp.create_approved_ticket.assert_not_called()
    assert app_module.app.state.ticket_store.list_ticket_references() == []


def test_test_draft_reject_never_writes(client, fake_mcp):
    action_id = draft(client)
    assert approve(client, "s1", action_id, False).json()["state"] == "rejected"
    fake_mcp.create_approved_ticket.assert_not_called()
    assert app_module.app.state.ticket_store.list_ticket_references() == []


def test_test_draft_repeated_decisions(client, fake_mcp):
    a = draft(client, "s1")
    assert approve(client, "s1", a, True).json()["state"] == "approved"
    assert approve(client, "s1", a, True).json()["state"] == "already_approved"
    assert approve(client, "s1", a, False).json()["state"] == "decision_already_recorded"

    b = draft(client, "s2")
    assert approve(client, "s2", b, False).json()["state"] == "rejected"
    assert approve(client, "s2", b, False).json()["state"] == "already_rejected"
    assert approve(client, "s2", b, True).json()["state"] == "decision_already_recorded"
    fake_mcp.create_approved_ticket.assert_not_called()


# --- unknown ids and session mismatch ----------------------------------------


def test_unknown_action_id(client, fake_mcp):
    body = approve(client, "s1", "does-not-exist", True).json()
    assert body["state"] == "no_pending_action"
    assert body["result"] == {}
    fake_mcp.create_approved_ticket.assert_not_called()


def test_mismatched_session_id(client, fake_mcp, seed_ticket_action):
    seed_ticket_action("s1", "act-1")
    body = approve(client, "other-session", "act-1", True).json()
    assert body["state"] == "no_pending_action"
    fake_mcp.create_approved_ticket.assert_not_called()

    # The real owner can still act: the wrong session changed nothing.
    assert approve(client, "s1", "act-1", True).json()["state"] == "ticket_created"
    assert fake_mcp.create_approved_ticket.await_count == 1


def test_mismatched_session_for_test_draft(client, fake_mcp):
    action_id = draft(client, "s1")
    assert approve(client, "s2", action_id, True).json()["state"] == "no_pending_action"
    fake_mcp.create_approved_ticket.assert_not_called()


def test_approve_requires_fields(client):
    r = client.post("/api/approve", json={"session_id": "s1", "action_id": "a"})
    assert r.status_code == 422


# --- invalid payloads --------------------------------------------------------


def _without(key):
    return {k: v for k, v in VALID_PAYLOAD.items() if k != key}


@pytest.mark.parametrize(
    "payload",
    [
        _without("subject"),
        {**VALID_PAYLOAD, "priority": "P9"},
        {**VALID_PAYLOAD, "extra": "x"},
        {**VALID_PAYLOAD, "requester_email": "someone@example.com"},
        {**VALID_PAYLOAD, "attempted_steps": []},
        "not-a-dict",
        None,
    ],
)
def test_invalid_payload_not_written(client, fake_mcp, seed_ticket_action, payload):
    seed_ticket_action("s1", "act-bad", payload=payload)
    body = approve(client, "s1", "act-bad", True).json()
    assert body["state"] == "invalid_pending_action"
    assert body["result"] == {}
    fake_mcp.create_approved_ticket.assert_not_called()


def test_unsupported_action_type_not_written(client, fake_mcp, seed_ticket_action):
    seed_ticket_action("s1", "act-x", type="delete_everything")
    assert approve(client, "s1", "act-x", True).json()["state"] == "unsupported_action"
    fake_mcp.create_approved_ticket.assert_not_called()


# --- approved ticket creation (mocked writer) --------------------------------


def test_approval_writes_once(client, fake_mcp, seed_ticket_action):
    seed_ticket_action("s1", "act-1")
    body = approve(client, "s1", "act-1", True).json()
    assert body["state"] == "ticket_created"
    assert body["result"] == {"ticket_id": "IT-2001"}
    assert body["pending_action"] is None

    fake_mcp.create_approved_ticket.assert_awaited_once()
    authorization = fake_mcp.create_approved_ticket.await_args.args[0]
    assert authorization.action_id == "act-1"
    assert authorization.payload == VALID_PAYLOAD
    metadata = app_module.app.state.ticket_store.get_ticket_reference("act-1")
    assert metadata is not None
    assert metadata["external_ticket_id"] == "IT-2001"
    assert metadata["payload_sha256"] == payload_digest(VALID_PAYLOAD)


def test_repeated_approval_not_written_twice(client, fake_mcp, seed_ticket_action):
    seed_ticket_action("s1", "act-1")
    approve(client, "s1", "act-1", True)
    again = approve(client, "s1", "act-1", True).json()
    assert again["state"] == "already_approved"
    assert again["result"] == {"ticket_id": "IT-2001"}
    assert fake_mcp.create_approved_ticket.await_count == 1


def test_rejection_never_writes(client, fake_mcp, seed_ticket_action):
    seed_ticket_action("s1", "act-1")
    assert approve(client, "s1", "act-1", False).json()["state"] == "rejected"
    assert approve(client, "s1", "act-1", False).json()["state"] == "already_rejected"
    assert approve(client, "s1", "act-1", True).json()["state"] == "decision_already_recorded"
    fake_mcp.create_approved_ticket.assert_not_called()
    assert app_module.app.state.ticket_store.list_ticket_references() == []


def test_local_persistence_failure_does_not_repeat_remote_creation(
    client, fake_mcp, seed_ticket_action, monkeypatch
):
    seed_ticket_action("s1", "act-1")
    store = app_module.app.state.ticket_store
    record_created_ticket = store.record_created_ticket

    def fail_record(**kwargs):
        raise TicketStoreError("database unavailable")

    monkeypatch.setattr(store, "record_created_ticket", fail_record)
    first = approve(client, "s1", "act-1", True).json()
    assert first["state"] == "ticket_created_local_persistence_failed"
    assert first["result"] == {"ticket_id": "IT-2001"}
    fake_mcp.create_approved_ticket.assert_awaited_once()

    monkeypatch.setattr(
        store,
        "record_created_ticket",
        record_created_ticket,
    )
    repeated = approve(client, "s1", "act-1", True).json()
    assert repeated["state"] == "already_approved"
    assert repeated["result"] == {"ticket_id": "IT-2001"}
    fake_mcp.create_approved_ticket.assert_awaited_once()
    assert store.get_ticket_reference("act-1") is not None


def test_approved_then_reject_is_refused(client, fake_mcp, seed_ticket_action):
    seed_ticket_action("s1", "act-1")
    approve(client, "s1", "act-1", True)
    assert approve(client, "s1", "act-1", False).json()["state"] == "decision_already_recorded"
    assert fake_mcp.create_approved_ticket.await_count == 1


@pytest.mark.parametrize(
    "result",
    [
        CallToolResult(content=[TextContent(type="text", text="boom")], isError=True),
        CallToolResult(content=[], structuredContent={"ok": False, "error": {"message": "no"}}),
        CallToolResult(content=[], structuredContent={"ok": True, "ticket": {"ticket_id": "BAD"}}),
        CallToolResult(content=[TextContent(type="text", text="not json")]),
    ],
)
def test_writer_failure_reported_and_not_marked_created(
    client, fake_mcp, seed_ticket_action, result
):
    fake_mcp.create_approved_ticket.return_value = result
    seed_ticket_action("s1", "act-1")
    body = approve(client, "s1", "act-1", True).json()
    assert body["state"] == "ticket_creation_failed"
    assert body["result"] == {}


def test_writer_exception_reported(client, fake_mcp, seed_ticket_action):
    fake_mcp.create_approved_ticket.side_effect = RuntimeError("down")
    seed_ticket_action("s1", "act-1")
    assert approve(client, "s1", "act-1", True).json()["state"] == "ticket_creation_failed"


def test_failed_creation_can_be_retried_once_succeeded(client, fake_mcp, seed_ticket_action):
    fake_mcp.create_approved_ticket.side_effect = [RuntimeError("down"), ticket_result("IT-2002")]
    seed_ticket_action("s1", "act-1")
    assert approve(client, "s1", "act-1", True).json()["state"] == "ticket_creation_failed"
    retry = approve(client, "s1", "act-1", True).json()
    assert retry["state"] == "ticket_created"
    assert retry["result"] == {"ticket_id": "IT-2002"}
    assert approve(client, "s1", "act-1", True).json()["state"] == "already_approved"
    assert fake_mcp.create_approved_ticket.await_count == 2


# --- concurrency -------------------------------------------------------------


def test_concurrent_duplicate_approvals_write_once(fake_mcp, seed_ticket_action):
    app_module.app.state.pending_actions.clear()
    seed_ticket_action("s1", "act-1")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_writer(authorization):
        entered.set()
        await release.wait()
        return ticket_result()

    fake_mcp.create_approved_ticket = AsyncMock(side_effect=slow_writer)
    payload = {"session_id": "s1", "action_id": "act-1", "approved": True}

    async def scenario():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            first = asyncio.create_task(ac.post("/api/approve", json=payload))
            await asyncio.wait_for(entered.wait(), timeout=5)
            # First write is still running; duplicates must not start another.
            duplicates = [await ac.post("/api/approve", json=payload) for _ in range(3)]
            release.set()
            return await first, duplicates

    try:
        first, duplicates = asyncio.run(scenario())
    finally:
        app_module.app.state.pending_actions.clear()

    assert first.json()["state"] == "ticket_created"
    assert {d.json()["state"] for d in duplicates} == {"creation_in_progress"}
    assert fake_mcp.create_approved_ticket.await_count == 1


def _stub_agent(monkeypatch, reply="I could not confirm a KB solution."):
    monkeypatch.setattr(
        app_module,
        "run_agent",
        AsyncMock(
            return_value={
                "reply": reply,
                "citations": [],
                "tool_calls": [],
                "pending_action": None,
            }
        ),
    )


def _ticket_request(client, message, email="dana.levi@nakhla.example"):
    return client.post(
        "/api/chat",
        json={
            "session_id": "draft-session",
            "message": message,
            "user": {"email": email},
        },
    )


def _complete_ticket_report():
    return (
        "Please create a support ticket. NakhlaConnect shows NX-781. "
        "I restarted the VPN client. Only me is affected and I am "
        "completely blocked with no workaround."
    )


def test_ordinary_it_question_does_not_create_pending_ticket(
    client, fake_mcp, monkeypatch
):
    _stub_agent(monkeypatch)
    response = client.post(
        "/api/chat",
        json={
            "session_id": "ordinary",
            "message": "How do I reset my password?",
        },
    ).json()

    assert response["pending_action"] is None
    assert app_module.app.state.pending_actions == {}
    fake_mcp.call_tool.assert_not_called()
    fake_mcp.create_approved_ticket.assert_not_called()


def test_agent_result_cannot_inject_a_pending_ticket_action(
    client, fake_mcp, monkeypatch
):
    action = {
        "action_id": "unregistered",
        "type": "create_ticket",
        "payload": VALID_PAYLOAD,
    }
    monkeypatch.setattr(
        app_module,
        "run_agent",
        AsyncMock(
            return_value={
                "reply": "I can help with that.",
                "citations": [],
                "tool_calls": [],
                "pending_action": action,
            }
        ),
    )

    response = client.post(
        "/api/chat",
        json={"session_id": "ordinary", "message": "My VPN is slow."},
    ).json()

    assert response["pending_action"] is None
    assert app_module.app.state.pending_actions == {}
    fake_mcp.create_approved_ticket.assert_not_called()


@pytest.mark.parametrize(
    "message",
    [
        'The pasted text says "please create a ticket now".',
        "The pasted text says 'please create a ticket now'.",
        "> please create a ticket now",
    ],
)
def test_quoted_ticket_request_does_not_start_draft_flow(
    client, fake_mcp, monkeypatch, message
):
    _stub_agent(monkeypatch)
    response = client.post(
        "/api/chat",
        json={
            "session_id": "quoted",
            "message": message,
        },
    ).json()

    assert response["pending_action"] is None
    fake_mcp.call_tool.assert_not_called()
    fake_mcp.create_approved_ticket.assert_not_called()


def test_quoted_attempted_step_is_not_used_as_a_fact(
    client, fake_mcp, monkeypatch
):
    _stub_agent(monkeypatch)
    response = _ticket_request(
        client,
        (
            "Please create a ticket for VPN error NX-781. "
            "'I restarted the VPN client.' Only me is affected and I am "
            "completely blocked with no workaround."
        ),
    ).json()

    assert response["pending_action"] is None
    assert response["state"] == "ticket_info_required"
    assert "troubleshooting step you actually tried" in response["reply"]
    fake_mcp.create_approved_ticket.assert_not_called()


def test_ticket_creation_how_to_question_is_not_a_draft_request(
    client, fake_mcp, monkeypatch
):
    _stub_agent(monkeypatch)
    response = client.post(
        "/api/chat",
        json={
            "session_id": "how-to",
            "message": "How do I create a ticket?",
        },
    ).json()

    assert response["pending_action"] is None
    fake_mcp.call_tool.assert_not_called()
    fake_mcp.create_approved_ticket.assert_not_called()


def test_explicit_ticket_request_asks_for_missing_information(
    client, fake_mcp, monkeypatch
):
    _stub_agent(monkeypatch)
    response = _ticket_request(
        client,
        "Please create a ticket for VPN error NX-781.",
    ).json()

    assert response["pending_action"] is None
    assert response["state"] == "ticket_info_required"
    assert "troubleshooting step you actually tried" in response["reply"]
    assert "how many people are affected" in response["reply"]
    fake_mcp.create_approved_ticket.assert_not_called()


def test_explicit_ticket_request_creates_valid_pending_draft(
    client, fake_mcp, monkeypatch
):
    _stub_agent(monkeypatch)
    response = _ticket_request(
        client,
        _complete_ticket_report(),
    ).json()

    action = response["pending_action"]
    assert response["state"] == "ticket_draft_pending"
    assert action["type"] == "create_ticket"
    assert app_module._ticket_payload_errors(action["payload"]) == []
    assert action["payload"]["requester_email"] == "dana.levi@nakhla.example"
    assert action["payload"]["department"] == "Engineering"
    assert action["payload"]["attempted_steps"] == [
        "restarted the VPN client"
    ]
    assert "no ticket has been created" in response["reply"].lower()
    assert fake_mcp.create_approved_ticket.await_count == 0
    assert app_module.app.state.ticket_store.list_ticket_references() == []


def test_ticket_intake_accepts_follow_up_without_repeating_request(
    client, fake_mcp, monkeypatch
):
    _stub_agent(monkeypatch)
    first = _ticket_request(
        client,
        "Please create a ticket for VPN error NX-781.",
    ).json()
    assert first["pending_action"] is None

    follow_up = client.post(
        "/api/chat",
        json={
            "session_id": "draft-session",
            "message": (
                "I restarted the VPN client. Only me is affected and "
                "I am completely blocked with no workaround."
            ),
        },
    ).json()

    assert follow_up["state"] == "ticket_draft_pending"
    assert app_module._ticket_payload_errors(
        follow_up["pending_action"]["payload"]
    ) == []
    fake_mcp.create_approved_ticket.assert_not_called()


def test_unverified_employee_cannot_receive_ticket_draft(
    client, fake_mcp, monkeypatch
):
    _stub_agent(monkeypatch)

    async def unknown_employee(name, arguments=None):
        if name == "lookup_employee":
            return CallToolResult(
                content=[],
                structuredContent={
                    "ok": False,
                    "error": {"code": "employee_not_found"},
                },
                isError=False,
            )
        return await fake_mcp._read_tool(name, arguments)

    fake_mcp.call_tool.side_effect = unknown_employee

    response = _ticket_request(
        client,
        _complete_ticket_report(),
        email="unknown@nakhla.example",
    ).json()

    assert response["pending_action"] is None
    assert response["state"] == "requester_unverified"
    fake_mcp.create_approved_ticket.assert_not_called()


def test_active_service_incident_prevents_individual_ticket(
    client, fake_mcp, monkeypatch
):
    _stub_agent(monkeypatch)
    active_status = CallToolResult(
        content=[],
        structuredContent={
            "ok": True,
            "status": {
                "services": [{"id": "vpn", "status": "degraded"}],
                "incidents": [
                    {
                        "id": "INC-2207",
                        "status": "active",
                        "service": "vpn",
                        "error_codes": ["NX-503"],
                        "workaround": "Select gateway GW-1 and reconnect.",
                        "instructions": "Subscribe to INC-2207 for updates.",
                    }
                ],
            },
        },
        isError=False,
    )

    async def active_status_tool(name, arguments=None):
        if name == "get_service_status":
            return active_status
        return await fake_mcp._read_tool(name, arguments)

    fake_mcp.call_tool.side_effect = active_status_tool
    report = (
        "Please create a ticket. NakhlaConnect shows NX-503. "
        "I switched gateways and retried. Only me is affected."
    )

    response = _ticket_request(client, report).json()

    assert response["pending_action"] is None
    assert response["state"] == "active_incident"
    assert "INC-2207" in response["reply"]
    assert "Select gateway GW-1" in response["reply"]
    called = [call.args[0] for call in fake_mcp.call_tool.await_args_list]
    assert called.count("get_service_status") == 1
    assert called.count("lookup_employee") == 1
    assert set(called) <= {"get_service_status", "lookup_employee"}
    fake_mcp.create_approved_ticket.assert_not_called()


def test_contractor_vpn_problem_uses_vdi_policy_before_status_workaround(
    client, fake_mcp, monkeypatch
):
    _stub_agent(monkeypatch)
    active_status = CallToolResult(
        content=[],
        structuredContent={
            "ok": True,
            "status": {
                "services": [{"id": "vpn", "status": "degraded"}],
                "incidents": [
                    {
                        "id": "INC-2207",
                        "status": "active",
                        "service": "vpn",
                        "error_codes": ["NX-503"],
                        "workaround": "Select gateway GW-1 and reconnect.",
                    }
                ],
            },
        },
        isError=False,
    )

    async def contractor_and_status(name, arguments=None):
        if name == "get_service_status":
            return active_status
        email = (arguments or {}).get("email", "contractor@nakhla.example")
        return CallToolResult(
            content=[],
            structuredContent={
                "ok": True,
                "employee": {
                    "email": email,
                    "department": "Contractors",
                },
            },
            isError=False,
        )

    fake_mcp.call_tool.side_effect = contractor_and_status

    response = _ticket_request(
        client,
        "Please create a ticket for VPN access. I tried reconnecting.",
        email="contractor@nakhla.example",
    ).json()

    assert response["state"] == "ticket_not_applicable"
    assert response["pending_action"] is None
    assert "VDI portal" in response["reply"]
    assert "gateway GW-1" not in response["reply"]
    fake_mcp.create_approved_ticket.assert_not_called()


def test_invalid_generated_ticket_payload_is_not_registered(
    client, fake_mcp, monkeypatch
):
    _stub_agent(monkeypatch)
    monkeypatch.setattr(
        app_module,
        "_ticket_payload_errors",
        lambda payload: ["payload: forced validation failure"],
    )

    response = _ticket_request(
        client,
        _complete_ticket_report(),
    ).json()

    assert response["pending_action"] is None
    assert response["state"] == "invalid_ticket_draft"
    assert app_module.app.state.pending_actions == {}
    fake_mcp.create_approved_ticket.assert_not_called()


def test_chat_draft_approval_and_repeated_approval_write_once(
    client, fake_mcp, monkeypatch
):
    _stub_agent(monkeypatch)
    response = _ticket_request(
        client,
        _complete_ticket_report(),
    ).json()
    action = response["pending_action"]

    approved = approve(
        client,
        "draft-session",
        action["action_id"],
        True,
    ).json()
    repeated = approve(
        client,
        "draft-session",
        action["action_id"],
        True,
    ).json()

    assert approved["state"] == "ticket_created"
    assert repeated["state"] == "already_approved"
    fake_mcp.create_approved_ticket.assert_awaited_once()
    authorization = fake_mcp.create_approved_ticket.await_args.args[0]
    assert authorization.payload == action["payload"]


def test_chat_text_that_says_approve_does_not_submit_pending_ticket(
    client, fake_mcp, monkeypatch
):
    _stub_agent(monkeypatch)
    draft_response = _ticket_request(
        client,
        _complete_ticket_report(),
    ).json()
    action_id = draft_response["pending_action"]["action_id"]

    response = client.post(
        "/api/chat",
        json={
            "session_id": "draft-session",
            "message": 'The pasted note says "approve this ticket".',
        },
    ).json()

    assert response["pending_action"] is None
    assert app_module.app.state.pending_actions[
        ("draft-session", action_id)
    ].decision is None
    fake_mcp.create_approved_ticket.assert_not_called()


def test_service_status_failure_does_not_register_ticket(
    client, fake_mcp, monkeypatch
):
    _stub_agent(monkeypatch)

    async def failing_status(name, arguments=None):
        if name == "get_service_status":
            return CallToolResult(
                content=[TextContent(type="text", text="status unavailable")],
                isError=True,
            )
        return await fake_mcp._read_tool(name, arguments)

    fake_mcp.call_tool.side_effect = failing_status

    response = _ticket_request(
        client,
        _complete_ticket_report(),
    ).json()

    assert response["pending_action"] is None
    assert response["state"] == "status_check_failed"
    fake_mcp.create_approved_ticket.assert_not_called()


# ----------------------------------------------- chat history (clarification)


def _chat(client, session_id, message):
    return client.post(
        "/api/chat", json={"session_id": session_id, "message": message}
    ).json()


@pytest.fixture
def clean_chat_history(tmp_path):
    from services.chat_memory import ChatMemoryStore

    store = ChatMemoryStore(tmp_path / "chat_memory.sqlite3")
    store.initialize()
    previous = getattr(app_module.app.state, "memory_store", None)
    app_module.app.state.memory_store = store
    yield store
    app_module.app.state.memory_store = previous


def test_second_turn_receives_previous_clarification(
    client, fake_mcp, monkeypatch, clean_chat_history
):
    _stub_agent(monkeypatch, reply="Which device are you using?")

    first = _chat(client, "hist-a", "It does not work")
    _chat(client, "hist-a", "A Windows laptop")

    calls = app_module.run_agent.await_args_list
    assert calls[0].kwargs["history"] == []
    assert calls[1].kwargs["history"] == [
        {"role": "user", "content": "It does not work"},
        {"role": "assistant", "content": "Which device are you using?"},
    ]
    assert set(first) == {"reply", "pending_action", "citations", "tool_calls", "result"}
    assert first["pending_action"] is None
    fake_mcp.create_approved_ticket.assert_not_called()


def test_history_is_isolated_between_sessions(
    client, fake_mcp, monkeypatch, clean_chat_history
):
    _stub_agent(monkeypatch)

    _chat(client, "hist-a", "My VPN fails")
    _chat(client, "hist-b", "My printer is jammed")

    assert app_module.run_agent.await_args_list[1].kwargs["history"] == []


def test_history_is_bounded(client, fake_mcp, monkeypatch, clean_chat_history):
    from services.chat_memory import MAX_STORED_MESSAGES

    _stub_agent(monkeypatch)

    for number in range(30):
        _chat(client, "hist-a", f"message {number}")

    history = app_module.run_agent.await_args_list[-1].kwargs["history"]
    assert len(history) == app_module.MAX_CHAT_HISTORY_MESSAGES
    assert history[-2]["content"] == "message 28"
    assert len(clean_chat_history.history("hist-a", limit=1000)) <= MAX_STORED_MESSAGES
