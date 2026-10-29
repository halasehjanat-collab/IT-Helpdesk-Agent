"""Draft -> approve/reject -> persisted -> listed/counted, fully offline.

A stateful in-memory fake stands in for the helpdesk system of record, so the count
can be observed changing without touching any real database or API.
"""

import pytest
from mcp.types import CallToolResult

from tests.conftest import VALID_PAYLOAD, ticket_result

EMAIL = VALID_PAYLOAD["requester_email"]


@pytest.fixture
def system(fake_mcp):
    """Wire the fake so create_approved_ticket stores a ticket that search_tickets returns."""
    tickets: list[dict] = []
    original_read = fake_mcp._read_tool

    async def create(authorization):
        ticket_id = f"IT-{3000 + len(tickets)}"
        tickets.append(
            {
                "ticket_id": ticket_id,
                "subject": authorization.payload["subject"],
                "category": authorization.payload["category"],
                "priority": authorization.payload["priority"],
                "status": "open",
                "department": authorization.payload["department"],
                "created_at": "2025-01-01T00:00:00Z",
                "requester_email": authorization.payload["requester_email"],
                "description": "private body that must not be listed",
            }
        )
        return ticket_result(ticket_id)

    async def read(name, arguments=None):
        if name == "search_tickets":
            mine = [t for t in tickets if t["requester_email"] == arguments["requester_email"]]
            return CallToolResult(
                content=[],
                structuredContent={
                    "ok": True,
                    "tickets": mine,
                    "count": len(mine),
                    "total_matches": len(mine),
                },
                isError=False,
            )
        return await original_read(name, arguments)

    fake_mcp.create_approved_ticket.side_effect = create
    fake_mcp.call_tool.side_effect = read
    return tickets


def _list(client, email=EMAIL):
    return client.get("/api/tickets", params={"requester_email": email})


def _approve(client, approved=True, session="s1", action="act-1"):
    return client.post(
        "/api/approve",
        json={"session_id": session, "action_id": action, "approved": approved},
    )


def test_list_is_empty_before_any_ticket(client, system):
    body = _list(client).json()
    assert body["state"] == "ok"
    assert body["count"] == 0 and body["total"] == 0


def test_pending_draft_does_not_change_count(client, system, seed_ticket_action, fake_mcp):
    seed_ticket_action()
    assert _list(client).json()["count"] == 0
    fake_mcp.create_approved_ticket.assert_not_awaited()


def test_approval_creates_one_ticket_and_count_rises_by_one(
    client, system, seed_ticket_action, fake_mcp
):
    seed_ticket_action()
    resp = _approve(client)
    body = resp.json()
    assert body["state"] == "ticket_created"
    assert body["result"]["ticket_id"] == "IT-3000"

    listed = _list(client).json()
    assert listed["count"] == 1 and listed["total"] == 1
    assert listed["tickets"][0]["ticket_id"] == "IT-3000"
    assert fake_mcp.create_approved_ticket.await_count == 1


def test_repeated_approval_does_not_increase_count(client, system, seed_ticket_action):
    seed_ticket_action()
    _approve(client)
    _approve(client)
    _approve(client)
    assert _list(client).json()["count"] == 1


def test_rejection_creates_nothing_and_count_unchanged(
    client, system, seed_ticket_action, fake_mcp
):
    seed_ticket_action()
    assert _approve(client, approved=False).json()["state"] == "rejected"
    assert _list(client).json()["count"] == 0
    fake_mcp.create_approved_ticket.assert_not_awaited()


def test_failed_write_is_not_reported_as_success_and_not_counted(
    client, system, seed_ticket_action, fake_mcp
):
    seed_ticket_action()
    fake_mcp.create_approved_ticket.side_effect = RuntimeError("boom")
    body = _approve(client).json()
    assert body["state"] != "ticket_created"
    assert "ticket_id" not in (body.get("result") or {})
    assert _list(client).json()["count"] == 0


def test_list_exposes_only_minimal_fields(client, system, seed_ticket_action):
    seed_ticket_action()
    _approve(client)
    ticket = _list(client).json()["tickets"][0]
    assert "description" not in ticket and "requester_email" not in ticket
    assert set(ticket) == {
        "ticket_id", "subject", "category", "priority", "status", "department", "created_at",
    }


def test_list_never_calls_the_write_function(client, system, fake_mcp):
    _list(client)
    fake_mcp.create_approved_ticket.assert_not_awaited()
    names = [c.args[0] for c in fake_mcp.call_tool.await_args_list]
    assert "create_ticket" not in names


@pytest.mark.parametrize("email", ["not-an-email", "someone@gmail.com"])
def test_unverified_requester_cannot_list(client, system, email):
    resp = _list(client, email)
    assert resp.status_code == 403
    assert resp.json()["state"] == "requester_not_verified"


def test_list_does_not_leak_other_requesters_tickets(client, system):
    system.append(
        {"ticket_id": "IT-1", "subject": "x", "requester_email": "other@nakhla.example"}
    )
    assert _list(client).json()["count"] == 0


def test_backend_failure_returns_safe_error(client, system, fake_mcp):
    original = fake_mcp.call_tool.side_effect

    async def broken(name, arguments=None):
        if name == "search_tickets":
            raise RuntimeError("down")
        return await original(name, arguments)

    fake_mcp.call_tool.side_effect = broken
    resp = _list(client)
    assert resp.status_code == 502
    assert resp.json() == {"state": "tickets_unavailable"}


def test_malformed_tool_response_is_rejected(client, system, fake_mcp):
    original = fake_mcp.call_tool.side_effect

    async def malformed(name, arguments=None):
        if name == "search_tickets":
            return CallToolResult(
                content=[], structuredContent={"ok": True, "tickets": "nope"}, isError=False
            )
        return await original(name, arguments)

    fake_mcp.call_tool.side_effect = malformed
    assert _list(client).status_code == 502
