"""Regression tests for ticket intake identity, classification and live status."""

from __future__ import annotations

from mcp.types import CallToolResult

import app as app_module


def _employee_result(email, *, name="Tariq Hamdan", department="Engineering"):
    return CallToolResult(
        content=[],
        structuredContent={
            "ok": True,
            "employee": {
                "email": email,
                "name": name,
                "department": department,
            },
        },
        isError=False,
    )


def _status_result(*, services=None, incidents=None):
    return CallToolResult(
        content=[],
        structuredContent={
            "ok": True,
            "status": {
                "services": services or [],
                "incidents": incidents or [],
            },
        },
        isError=False,
    )


def _set_directory_and_status(fake_mcp, employee_result):
    async def lookup_and_status(name, arguments=None):
        if name == "lookup_employee":
            return employee_result
        if name == "get_service_status":
            return _status_result()
        return await fake_mcp._read_tool(name, arguments)

    fake_mcp.call_tool.side_effect = lookup_and_status


def _ticket_request(client, session_id, message, email):
    return client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": message,
            "user": {"email": email},
        },
    ).json()


def test_network_ticket_uses_issue_subject_category_steps_impact_and_policy_refs(
    client, fake_mcp
):
    email = "tariq.hamdan@nakhla.example"
    _set_directory_and_status(fake_mcp, _employee_result(email))
    message = (
        "Hi, I cannot access the company network. "
        "I restarted the computer and retried the connection. "
        "I am the only affected user, my work is completely blocked, "
        "and I have no workaround. Please create a ticket."
    )

    response = _ticket_request(client, "network-intake-regression", message, email)
    action = response["pending_action"]
    payload = action["payload"]

    assert response["state"] == "ticket_draft_pending"
    assert payload["subject"] == "Unable to access the company network"
    assert payload["category"] == "network_access"
    assert payload["priority"] == "P2"
    assert payload["affected_users"] == 1
    assert payload["attempted_steps"] == [
        "restarted the computer",
        "retried the connection",
    ]
    assert "KB-015" in payload["kb_refs"]
    assert "work blocked" not in response["reply"].lower() or "priority p3" not in response["reply"].lower()
    fake_mcp.create_approved_ticket.assert_not_awaited()


def test_blocked_work_without_workaround_information_requests_clarification(
    client, fake_mcp
):
    email = "tariq.hamdan@nakhla.example"
    _set_directory_and_status(fake_mcp, _employee_result(email))
    message = (
        "Please create a ticket. I cannot access the company network. "
        "I restarted the computer. I am the only affected user and my work is blocked."
    )

    response = _ticket_request(client, "missing-workaround", message, email)

    assert response["state"] == "ticket_info_required"
    assert response["pending_action"] is None
    assert "workaround" in response["reply"].lower()
    assert fake_mcp.create_approved_ticket.await_count == 0


def test_work_blocked_with_no_workaround_is_extracted_from_natural_wording(
    client, fake_mcp
):
    email = "tariq.hamdan@nakhla.example"
    _set_directory_and_status(fake_mcp, _employee_result(email))
    message = (
        "Please prepare a support ticket. I cannot access the company network. "
        "I restarted my laptop and retried the connection once. I am the only "
        "affected user, my work is blocked, and there is no workaround."
    )

    response = _ticket_request(client, "natural-impact-wording", message, email)
    action = response["pending_action"]

    assert response["state"] == "ticket_draft_pending"
    assert action["payload"]["priority"] == "P2"
    assert action["payload"]["affected_users"] == 1
    assert action["payload"]["attempted_steps"] == [
        "restarted my laptop",
        "retried the connection once",
    ]
    fake_mcp.create_approved_ticket.assert_not_awaited()


def test_priority_recognizes_work_blocked_by_an_installation_failure():
    assert app_module._ticket_priority(
        "The failed installation completely blocks my work, and I have no workaround.",
        "Engineering",
        1,
    ) == "P2"


def test_requester_name_conflicting_with_directory_is_not_used_for_draft(
    client, fake_mcp
):
    email = "tariq.hamdan@nakhla.example"
    _set_directory_and_status(fake_mcp, _employee_result(email))
    message = (
        "Please create a ticket. Hi, I'm Sara Mikhail from Finance. "
        "I cannot access the company network. I restarted my computer. "
        "I am the only affected user, completely blocked, no workaround."
    )

    response = _ticket_request(client, "name-mismatch", message, email)

    assert response["state"] == "requester_identity_mismatch"
    assert response["pending_action"] is None
    assert "work email" in response["reply"].lower()
    fake_mcp.create_approved_ticket.assert_not_awaited()


def test_requester_department_conflicting_with_directory_is_not_used_for_draft(
    client, fake_mcp
):
    email = "tariq.hamdan@nakhla.example"
    _set_directory_and_status(fake_mcp, _employee_result(email))
    message = (
        "Please create a ticket. I am from Finance. "
        "I cannot access the company network. I restarted my computer. "
        "I am the only affected user, completely blocked, no workaround."
    )

    response = _ticket_request(client, "department-mismatch", message, email)

    assert response["state"] == "requester_identity_mismatch"
    assert response["pending_action"] is None
    fake_mcp.create_approved_ticket.assert_not_awaited()


def test_contractor_claim_conflicting_with_verified_finance_identity_is_clarified(
    client, fake_mcp, monkeypatch
):
    email = "rana.khalil@nakhla.example"
    _set_directory_and_status(
        fake_mcp,
        _employee_result(email, name="Rana Khalil", department="Finance"),
    )
    monkeypatch.setattr(
        app_module,
        "run_agent",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Identity conflict should be clarified before answering.")
        ),
    )

    response = client.post(
        "/api/chat",
        json={
            "session_id": "contractor-finance-conflict",
            "message": "I'm a contractor and need VPN access. How do I set it up?",
            "user": {"email": email},
        },
    ).json()

    assert response["state"] == "identity_clarification_required"
    assert "contractor" in response["reply"].lower()
    assert "finance" in response["reply"].lower()
    assert response["pending_action"] is None
    fake_mcp.create_approved_ticket.assert_not_awaited()


def test_requester_email_conflicting_with_ui_email_is_not_silently_replaced(
    client, fake_mcp
):
    email = "tariq.hamdan@nakhla.example"
    _set_directory_and_status(fake_mcp, _employee_result(email))
    message = (
        "Please create a ticket for sara.mikhail@nakhla.example. "
        "I cannot access the company network. I restarted my computer. "
        "I am the only affected user, completely blocked, no workaround."
    )

    response = _ticket_request(client, "email-mismatch", message, email)

    assert response["state"] == "requester_identity_mismatch"
    assert response["pending_action"] is None
    fake_mcp.create_approved_ticket.assert_not_awaited()


def test_explicit_vpn_status_question_uses_live_status_tool(
    client, fake_mcp, monkeypatch
):
    async def active_status(name, arguments=None):
        if name == "get_service_status":
            return _status_result(
                services=[{"id": "vpn", "name": "NakhlaConnect VPN", "status": "degraded"}],
                incidents=[
                    {
                        "id": "INC-2207",
                        "service": "vpn",
                        "status": "active",
                        "title": "VPN gateway GW-2 degraded",
                        "error_codes": ["NX-503"],
                        "workaround": "Select GW-1 and reconnect.",
                    }
                ],
            )
        return await fake_mcp._read_tool(name, arguments)

    fake_mcp.call_tool.side_effect = active_status
    monkeypatch.setattr(
        app_module,
        "run_agent",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("A live status question should not use the LLM.")
        ),
    )

    response = client.post(
        "/api/chat",
        json={
            "session_id": "vpn-status-question",
            "message": "Can you check the current VPN status and active incident?",
        },
    ).json()

    assert "degraded" in response["reply"].lower()
    assert "INC-2207" in response["reply"]
    assert "NX-503" in response["reply"]
    assert response["pending_action"] is None
    assert [call.args[0] for call in fake_mcp.call_tool.await_args_list] == [
        "get_service_status"
    ]
    fake_mcp.create_approved_ticket.assert_not_awaited()


def test_live_status_tool_failure_does_not_claim_a_service_state(
    client, fake_mcp, monkeypatch
):
    async def failing_status(name, arguments=None):
        if name == "get_service_status":
            return CallToolResult(
                content=[],
                structuredContent={"ok": False, "error": {"code": "unreachable"}},
                isError=False,
            )
        return await fake_mcp._read_tool(name, arguments)

    fake_mcp.call_tool.side_effect = failing_status
    monkeypatch.setattr(
        app_module,
        "run_agent",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("A live status question should not use the LLM.")
        ),
    )

    response = client.post(
        "/api/chat",
        json={
            "session_id": "vpn-status-failure",
            "message": "Check the live VPN service status and whether there's an incident.",
        },
    ).json()

    assert "could not check" in response["reply"].lower()
    assert "operational" not in response["reply"].lower()
    assert "active incident" not in response["reply"].lower()
    assert response["pending_action"] is None
    fake_mcp.create_approved_ticket.assert_not_awaited()


def test_mfa_disable_conflict_gets_grounded_current_policy_answer(monkeypatch):
    import asyncio

    import agent as agent_module
    from services import answer_verifier

    monkeypatch.setattr(
        agent_module,
        "generate_reply_with_status",
        lambda *args, **kwargs: (answer_verifier.ABSTENTION, None, True),
    )
    response = asyncio.run(
        agent_module.run_agent(
            "An old guide says to disable MFA, but the current security policy "
            "says never to disable MFA. What should I do?"
        )
    )

    assert "MFA is mandatory" in response["reply"]
    assert "overrides any knowledge-base article" in response["reply"]
    assert "SP-002" in {citation["id"] for citation in response["citations"]}


def test_nid423_answer_includes_both_documented_unlock_options(monkeypatch):
    import asyncio

    import agent as agent_module

    monkeypatch.setattr(
        agent_module,
        "generate_reply_with_status",
        lambda *args, **kwargs: (
            "NID-423 means the account locked after failed sign-ins. [KB-006]",
            None,
            True,
        ),
    )
    response = asyncio.run(
        agent_module.run_agent(
            "NakhlaID says NID-423 when I try to sign in. What should I do?"
        )
    )

    assert "30 minutes" in response["reply"]
    assert "/unlock" in response["reply"]
    assert "KB-006" in {citation["id"] for citation in response["citations"]}
