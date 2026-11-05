"""Long-term memory across sessions, using a real (temporary) SQLite store."""
import pytest

import app as app_module
from services.chat_memory import ChatMemoryStore, extract_facts

TARIQ = "tariq.hamdan@nakhla.example"
DANA = "dana.levi@nakhla.example"
FIRST = (
    "My employee ID is EMP-1042, and I previously had a VPN connection issue. "
    "I restarted my computer, but the problem continued."
)
RECALL = (
    "Do you remember the IT issue I reported in our previous session? "
    "Please summarize what you remember about it and tell me what solution we tried."
)


@pytest.fixture
def memory_client(client, tmp_path):
    store = ChatMemoryStore(tmp_path / "ltm.sqlite3")
    store.initialize()
    previous = getattr(app_module.app.state, "memory_store", None)
    app_module.app.state.memory_store = store
    yield client
    app_module.app.state.memory_store = previous


def _chat(client, message, email, session):
    body = {"session_id": session, "message": message, "user": {"email": email}}
    return client.post("/api/chat", json=body).json()


def test_employee_id_is_extracted_as_fact():
    assert ("employee_id", "EMP-1042", "explicit") in extract_facts(FIRST)


def test_issue_saved_in_one_session_is_recalled_in_a_new_one(memory_client):
    _chat(memory_client, FIRST, TARIQ, "session-A")
    body = _chat(memory_client, RECALL, TARIQ, "session-B")

    reply = body["reply"].lower()
    assert "vpn" in reply and "restarted my computer" in reply
    assert body["pending_action"] is None


def test_memory_is_not_shared_between_users(memory_client):
    _chat(memory_client, FIRST, TARIQ, "session-A")
    body = _chat(memory_client, RECALL, DANA, "session-C")

    assert "vpn" not in body["reply"].lower()
    assert "EMP-1042" not in body["reply"]


def test_nothing_is_invented_when_nothing_was_saved(memory_client):
    body = _chat(memory_client, RECALL, TARIQ, "session-new")

    assert "vpn" not in body["reply"].lower()
    assert "don't have" in body["reply"].lower()


def test_recall_never_creates_a_ticket(memory_client, monkeypatch):
    called = []
    monkeypatch.setattr(
        app_module, "create_approved_ticket", lambda *a, **k: called.append(1), raising=False
    )
    _chat(memory_client, FIRST, TARIQ, "session-A")
    _chat(memory_client, RECALL, TARIQ, "session-B")
    assert called == []
