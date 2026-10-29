import sqlite3

import app as app_module
from services.approval_store import SQLiteApprovalStore
from tests.conftest import VALID_PAYLOAD, ticket_result


def approve(client, session="s1", action="act-1", approved=True):
    return client.post("/api/approve", json={"session_id": session, "action_id": action, "approved": approved})


def restart():
    """Drop in-memory state while keeping the SQLite approval store, like an app restart."""
    app_module.app.state.pending_actions.clear()


def test_pending_action_recovers_after_restart_and_creates_once(client, fake_mcp, seed_ticket_action):
    seed_ticket_action()
    restart()
    first = approve(client)
    assert first.json()["state"] == "ticket_created"
    restart()
    again = approve(client)
    assert again.json()["state"] == "already_approved"
    assert fake_mcp.create_approved_ticket.await_count == 1


def test_other_session_cannot_approve_restored_action(client, fake_mcp, seed_ticket_action):
    seed_ticket_action()
    restart()
    response = approve(client, session="intruder")
    assert response.json()["state"] == "no_pending_action"
    fake_mcp.create_approved_ticket.assert_not_awaited()


def test_rejection_survives_restart(client, fake_mcp, seed_ticket_action):
    seed_ticket_action()
    assert approve(client, approved=False).json()["state"] == "rejected"
    restart()
    assert approve(client, approved=True).json()["state"] == "decision_already_recorded"
    fake_mcp.create_approved_ticket.assert_not_awaited()


def test_failed_attempt_is_retryable_after_restart_with_same_action(client, fake_mcp, seed_ticket_action):
    seed_ticket_action()
    fake_mcp.create_approved_ticket.side_effect = [RuntimeError("down"), ticket_result("IT-2005")]
    assert approve(client).json()["state"] == "ticket_creation_failed"
    restart()
    assert approve(client).json()["state"] == "ticket_created"
    authorizations = [c.args[0] for c in fake_mcp.create_approved_ticket.await_args_list]
    assert {a.action_id for a in authorizations} == {"act-1"}


def test_interrupted_in_progress_action_loads_as_retryable(tmp_path):
    store = SQLiteApprovalStore(tmp_path / "a.sqlite3")
    store.initialize()
    action = {"action_id": "a", "type": "create_ticket", "summary": "s", "payload": dict(VALID_PAYLOAD)}
    store.save("sess", action, decision=True, write_status="in_progress", ticket_id=None, local_recorded=False)
    loaded = store.load("sess", "a")
    assert loaded["write_status"] == "failed" and loaded["decision"] is True


def test_decision_cannot_be_overwritten_and_session_is_hashed(tmp_path):
    path = tmp_path / "a.sqlite3"
    store = SQLiteApprovalStore(path)
    store.initialize()
    action = {"action_id": "a", "type": "create_ticket", "summary": "s", "payload": dict(VALID_PAYLOAD)}
    store.save("secret-session", action, decision=False, write_status="not_started", ticket_id=None, local_recorded=False)
    store.save("secret-session", action, decision=True, write_status="not_started", ticket_id=None, local_recorded=False)
    assert store.load("secret-session", "a")["decision"] is False
    raw = sqlite3.connect(path).execute("SELECT session_hash FROM approval_actions").fetchone()[0]
    assert raw != "secret-session" and b"secret-session" not in path.read_bytes()


def test_tampered_payload_is_refused(tmp_path):
    path = tmp_path / "a.sqlite3"
    store = SQLiteApprovalStore(path)
    store.initialize()
    action = {"action_id": "a", "type": "create_ticket", "summary": "s", "payload": dict(VALID_PAYLOAD)}
    store.save("sess", action, decision=None, write_status="not_started", ticket_id=None, local_recorded=False)
    connection = sqlite3.connect(path)
    connection.execute("UPDATE approval_actions SET action_json = replace(action_json, 'NX-781', 'EVIL')")
    connection.commit()
    connection.close()
    assert store.load("sess", "a") is None


def test_test_only_actions_are_never_persisted(client, fake_mcp, tmp_path):
    app_module._store_pending_action("s1", {"action_id": "t1", "type": "create_ticket", "test_only": True,
                                            "summary": "x", "payload": dict(VALID_PAYLOAD)})
    restart()
    assert approve(client, action="t1").json()["state"] == "no_pending_action"
    fake_mcp.create_approved_ticket.assert_not_awaited()


def test_created_ticket_draft_is_scrubbed_from_approval_store(client, fake_mcp, seed_ticket_action):
    seed_ticket_action()
    approve(client)
    store = app_module.app.state.approval_store
    row = sqlite3.connect(store.database_path).execute("SELECT action_json FROM approval_actions").fetchone()[0]
    assert VALID_PAYLOAD["description"] not in row
