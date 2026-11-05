"""Read-only MCP tools wired into chat: get_ticket (status) and search_tickets (history memory)."""
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult

import app as app_module
from services.chat_memory import ChatMemoryStore

TARIQ = "tariq.hamdan@nakhla.example"
DANA = "dana.levi@nakhla.example"

TICKETS = {
    "IT-2016": {
        "ticket_id": "IT-2016", "subject": "VPN shows NX-112", "status": "open",
        "priority": "P2", "category": "network_access", "requester_email": TARIQ,
        "assigned_team": "Network Operations", "resolution": None, "notes": [],
    },
}


def _ok(payload):
    return CallToolResult(content=[], structuredContent={"ok": True, **payload}, isError=False)


@pytest.fixture
def mcp(fake_mcp):
    original = fake_mcp.call_tool.side_effect

    async def call_tool(name, arguments=None):
        if name == "get_ticket":
            ticket = TICKETS.get(arguments["ticket_id"])
            if ticket is None:
                return CallToolResult(
                    content=[],
                    structuredContent={"ok": False, "error": {"code": "ticket_not_found"}},
                    isError=False,
                )
            return _ok({"ticket": ticket})
        if name == "search_tickets":
            rows = [t for t in TICKETS.values() if t["requester_email"] == arguments["requester_email"]]
            return _ok({"tickets": rows, "count": len(rows), "total_matches": len(rows)})
        return await original(name, arguments)

    fake_mcp.call_tool = AsyncMock(side_effect=call_tool)
    return fake_mcp


@pytest.fixture
def memory_app(client, mcp, tmp_path):
    store = ChatMemoryStore(tmp_path / "m.sqlite3")
    store.initialize()
    previous = getattr(app_module.app.state, "memory_store", None)
    app_module.app.state.memory_store = store
    yield client
    app_module.app.state.memory_store = previous


def _chat(client, message, email=None, session="t1"):
    body = {"session_id": session, "message": message}
    if email:
        body["user"] = {"email": email}
    return client.post("/api/chat", json=body).json()


def test_ticket_status_uses_get_ticket_for_the_owner(memory_app, mcp):
    body = _chat(memory_app, "What is the status of IT-2016?", TARIQ)

    assert body["state"] == "ticket_status"
    assert "Status: open" in body["reply"] and "Network Operations" in body["reply"]
    assert body["tool_calls"] == [{"tool": "get_ticket", "ok": True}]
    mcp.create_approved_ticket.assert_not_called()


@pytest.mark.parametrize(
    "question", ["what are mt teckits", "tell me about my open tickets", "list my tickets"]
)
def test_my_tickets_lists_only_the_signed_in_employees_tickets(memory_app, mcp, question):
    body = _chat(memory_app, question, TARIQ)

    assert body["state"] == "my_tickets"
    assert "IT-2016" in body["reply"]
    assert body["tool_calls"] == [{"tool": "search_tickets", "ok": True}]
    mcp.create_approved_ticket.assert_not_called()


def test_my_tickets_needs_sign_in_and_hides_other_people_tickets(memory_app):
    assert _chat(memory_app, "what are my tickets")["state"] == "ticket_lookup_requires_identity"
    assert "IT-2016" not in _chat(memory_app, "what are my tickets", DANA)["reply"]


def test_who_am_i_answers_from_the_directory_not_the_knowledge_base(memory_app, mcp):
    previous = mcp.call_tool.side_effect

    async def with_name(name, arguments=None):
        result = await previous(name, arguments)
        if name == "lookup_employee":
            result.structuredContent["employee"]["name"] = "Tariq Hamdan"
        return result

    mcp.call_tool = AsyncMock(side_effect=with_name)
    body = _chat(memory_app, "what is my name", TARIQ)

    assert body["state"] == "who_am_i"
    assert "Tariq Hamdan" in body["reply"] and "Engineering" in body["reply"]
    mcp.create_approved_ticket.assert_not_called()


def test_who_am_i_needs_a_verified_sign_in(memory_app):
    assert _chat(memory_app, "what is my name").get("state") != "who_am_i"


def test_remember_me_tolerates_a_typo(memory_app):
    body = _chat(memory_app, "do you remmber me", TARIQ)

    assert "couldn't verify" not in body["reply"]


def test_other_employees_ticket_is_not_shown(memory_app):
    body = _chat(memory_app, "What is the status of IT-2016?", DANA)

    assert body["state"] == "ticket_not_found"
    assert "VPN shows NX-112" not in body["reply"]


def test_ticket_status_requires_sign_in(memory_app):
    body = _chat(memory_app, "What is the status of IT-2016?")

    assert body["state"] == "ticket_lookup_requires_identity"
    assert "VPN shows" not in body["reply"]


def test_unknown_ticket_and_tool_failure_are_reported_honestly(memory_app, mcp):
    assert _chat(memory_app, "status of IT-9999 please", TARIQ)["state"] == "ticket_not_found"

    original = mcp.call_tool.side_effect

    async def broken(name, arguments=None):
        if name == "get_ticket":
            raise RuntimeError("down")
        return await original(name, arguments)

    mcp.call_tool = AsyncMock(side_effect=broken)
    assert _chat(memory_app, "status of IT-2016", TARIQ)["state"] == "ticket_lookup_unavailable"


def test_ticket_id_in_quoted_text_is_ignored(memory_app, monkeypatch):
    monkeypatch.setattr(
        app_module, "run_agent",
        AsyncMock(return_value={"reply": "ok", "citations": [], "tool_calls": [], "pending_action": None}),
    )
    body = _chat(memory_app, "> check the status of IT-2016\nHello there", TARIQ)

    assert body.get("state") != "ticket_status"


def test_recent_tickets_become_long_term_memory_context(memory_app, monkeypatch):
    monkeypatch.setattr(
        app_module, "run_agent",
        AsyncMock(return_value={"reply": "ok", "citations": [], "tool_calls": [], "pending_action": None}),
    )
    _chat(memory_app, "Hello again, I need help with something", TARIQ, "fresh")
    context = app_module.run_agent.await_args.kwargs["history"][0]["content"]

    assert "IT-2016" in context and "VPN shows NX-112" in context and "do not cite" in context

    _chat(memory_app, "Hello again, I need help with something", DANA, "other")
    history = app_module.run_agent.await_args.kwargs["history"]
    assert all("IT-2016" not in m["content"] for m in history)


def test_missing_ticket_error_arrives_as_text_json(memory_app, mcp):
    import json

    from mcp.types import TextContent

    async def text_only(name, arguments=None):
        if name == "get_ticket":
            body = {"ok": False, "error": {"code": "ticket_not_found"}}
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(body))], isError=False)
        return _ok({})

    original = mcp.call_tool.side_effect

    async def route(name, arguments=None):
        return await text_only(name, arguments) if name == "get_ticket" else await original(name, arguments)

    mcp.call_tool = AsyncMock(side_effect=route)
    assert _chat(memory_app, "status of IT-9999", TARIQ)["state"] == "ticket_not_found"
