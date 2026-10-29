"""Email format validation is separate from employee identity verification (offline)."""

import asyncio

import pytest

import app as app_module


@pytest.mark.parametrize(
    "email",
    [
        "user@gmail.com",
        "name@outlook.com",
        "employee@yahoo.com",
        "person@nakhla.example",
        "first.last+tag@sub.example.org",
    ],
)
def test_well_formed_addresses_pass_format_validation(email):
    assert app_module._is_valid_email_format(email)


@pytest.mark.parametrize(
    "email",
    ["user@", "@gmail.com", "user gmail.com", "user@@gmail.com", "a..b@gmail.com", "user@gmail", "", None],
)
def test_malformed_addresses_fail_format_validation(email):
    assert not app_module._is_valid_email_format(email)


@pytest.mark.parametrize(
    "message,expected",
    [
        ("my email is person@gmail.com thanks", "person@gmail.com"),
        ("Dana.Levi@Nakhla.Example", "dana.levi@nakhla.example"),
        ("no address here", None),
        ("broken user@ and @gmail.com", None),
    ],
)
def test_email_is_extracted_from_free_text_for_any_domain(message, expected):
    assert app_module._email_from_message(message) == expected


def _lookup(fake_mcp, email):
    app_module.app.state.mcp = fake_mcp
    return asyncio.run(app_module._lookup_employee(email))


def _reads(fake_mcp):
    return [c.args[0] for c in fake_mcp.call_tool.await_args_list]


def test_external_domain_is_not_treated_as_employee_identity(fake_mcp):
    # The fake directory would "verify" any address, so this proves the app itself refuses.
    employee, error = _lookup(fake_mcp, "user@gmail.com")

    assert employee is None
    assert error == app_module.EMAIL_NOT_VERIFIABLE_ERROR
    assert "valid email address" in error
    assert "lookup_employee" not in _reads(fake_mcp)


def test_malformed_address_gets_a_format_message(fake_mcp):
    employee, error = _lookup(fake_mcp, "user@")

    assert employee is None
    assert error == app_module.EMAIL_FORMAT_ERROR
    assert _reads(fake_mcp) == []


def test_nakhla_address_is_verified_through_the_directory(fake_mcp):
    employee, error = _lookup(fake_mcp, "dana.levi@nakhla.example")

    assert error is None
    assert employee["email"] == "dana.levi@nakhla.example"
    assert _reads(fake_mcp) == ["lookup_employee"]


def test_unknown_directory_record_is_not_verified(fake_mcp):
    async def mismatched(name, arguments=None):
        from mcp.types import CallToolResult

        return CallToolResult(
            content=[],
            structuredContent={
                "ok": True,
                "employee": {"email": "someone.else@nakhla.example", "department": "Engineering"},
            },
            isError=False,
        )

    fake_mcp.call_tool.side_effect = mismatched
    employee, error = _lookup(fake_mcp, "dana.levi@nakhla.example")

    assert employee is None
    assert error


def _post(client, message, email=None, session="s-email"):
    body = {"session_id": session, "message": message}
    if email:
        body["user"] = {"email": email}
    return client.post("/api/chat", json=body).json()


@pytest.mark.parametrize("email", ["user@gmail.com", "name@outlook.com", "employee@yahoo.com"])
def test_external_address_never_produces_a_draft_or_ticket(client, fake_mcp, email):
    body = _post(
        client,
        "Please prepare a ticket for my VPN problem. I restarted the VPN client.",
        email=email,
    )

    assert body["state"] == "requester_unverified"
    assert body["pending_action"] is None
    assert body["reply"].count("not a Nakhla work address") == 1
    assert body["reply"].count("Nakhla work email") == 1
    fake_mcp.create_approved_ticket.assert_not_called()


def test_malformed_address_gets_one_validation_message_and_no_draft(client, fake_mcp):
    body = _post(client, "Please prepare a ticket for my VPN problem", email="user@")

    assert body["state"] == "requester_unverified"
    assert body["pending_action"] is None
    assert body["reply"].count("valid email address") == 1
    assert body["reply"].count("Nakhla work email") == 1
    fake_mcp.create_approved_ticket.assert_not_called()


def test_verified_requester_advances_to_the_next_intake_step(client, fake_mcp):
    body = _post(
        client,
        "Please prepare a ticket for my VPN problem",
        email="dana.levi@nakhla.example",
    )

    # Identity is accepted, so the email is no longer asked for; the next missing item is.
    assert body["state"] == "ticket_info_required"
    assert "work email" not in body["reply"].lower()
    assert "troubleshooting step" in body["reply"]
    assert body["pending_action"] is None
    fake_mcp.create_approved_ticket.assert_not_called()


def test_completed_intake_gives_a_pending_draft_that_is_not_a_ticket(client, fake_mcp):
    body = _post(
        client,
        "Please prepare a ticket for my VPN problem. I restarted the VPN client and my laptop.",
        email="dana.levi@nakhla.example",
    )

    assert body["state"] == "ticket_draft_pending"
    action = body["pending_action"]
    assert action["payload"]["requester_email"] == "dana.levi@nakhla.example"
    assert "VPN" in action["payload"]["description"]
    assert "no ticket has been created" in body["reply"].lower()
    fake_mcp.create_approved_ticket.assert_not_called()


def test_pending_draft_cannot_be_created_without_approval(client, fake_mcp):
    body = _post(
        client,
        "Please prepare a ticket for my VPN problem. I restarted the VPN client and my laptop.",
        email="dana.levi@nakhla.example",
        session="s-approval",
    )
    assert body["pending_action"] is not None

    other = client.post(
        "/api/approve",
        json={"session_id": "another-session", "action_id": body["pending_action"]["action_id"], "approved": True},
    )

    assert other.json().get("state") != "ticket_created"
    fake_mcp.create_approved_ticket.assert_not_called()
