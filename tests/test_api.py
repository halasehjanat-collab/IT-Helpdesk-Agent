import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from mcp.types import CallToolResult, TextContent

import app as app_module
from tests.conftest import VALID_PAYLOAD, ticket_result

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


def test_test_draft_reject_never_writes(client, fake_mcp):
    action_id = draft(client)
    assert approve(client, "s1", action_id, False).json()["state"] == "rejected"
    fake_mcp.create_approved_ticket.assert_not_called()


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
