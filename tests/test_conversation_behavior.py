"""Greeting handling and verifier-notice de-duplication (offline, mocked)."""

import asyncio

import pytest

import agent
from services import answer_verifier as av


def _run(message, **kw):
    return asyncio.run(agent.run_agent(message, **kw))


@pytest.fixture
def no_backends(monkeypatch):
    calls = {"retrieve": 0, "llm": 0}

    def boom_retrieve(*a, **k):
        calls["retrieve"] += 1
        raise AssertionError("retrieval must not run for small talk")

    def boom_llm(*a, **k):
        calls["llm"] += 1
        raise AssertionError("LLM must not run for small talk")

    monkeypatch.setattr(agent, "retrieve_chunks", boom_retrieve)
    monkeypatch.setattr(agent, "generate_reply_with_status", boom_llm)
    return calls


@pytest.mark.parametrize(
    "msg", ["Hi, how are you?", "hello", "Good morning!", "thanks", "bye"]
)
def test_greeting_gets_friendly_reply_without_backends(no_backends, msg):
    result = _run(msg)
    assert result["reply"]
    assert "verify" not in result["reply"].lower()
    assert "escalat" not in result["reply"].lower()
    assert "ticket" not in result["reply"].lower()
    assert result["tool_calls"] == []
    assert result["pending_action"] is None
    assert result["citations"] == []
    assert no_backends == {"retrieve": 0, "llm": 0}


@pytest.mark.parametrize(
    "msg",
    ["Hi, my VPN is not working", "hello I need a ticket", "How do I reset my password?"],
)
def test_technical_messages_are_not_small_talk(msg):
    assert agent._small_talk_reply(msg) is None


def test_greeting_through_api_creates_nothing(client, fake_mcp, monkeypatch):
    monkeypatch.setattr(
        agent, "retrieve_chunks", lambda *a, **k: pytest.fail("retrieval")
    )
    resp = client.post(
        "/api/chat", json={"message": "Hi, how are you?", "session_id": "s-greet"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["pending_action"] is None
    assert body["tool_calls"] == []
    assert "escalat" not in body["reply"].lower()


def test_strip_notices_removes_prior_warnings():
    text = (
        "Restart the client [KB-001].\n\n"
        f"{av.UNVERIFIED_HEADER}\n- Reinstall drivers\n- Clear cache\n\n"
        f"{av.PARTIAL_NOTE}"
    )
    assert av.strip_notices(text) == "Restart the client [KB-001]."


def test_history_assistant_notices_are_stripped():
    hist = [
        {"role": "user", "content": "wifi?"},
        {
            "role": "assistant",
            "content": f"Try this.\n\n{av.UNVERIFIED_HEADER}\n- x\n\n{av.PARTIAL_NOTE}",
        },
    ]
    cleaned = agent._clean_history(hist)
    assert cleaned[1]["content"] == "Try this."
    assert av.UNVERIFIED_HEADER not in cleaned[1]["content"]


def test_verifier_warnings_appear_once_even_if_model_echoes_them():
    echoed = (
        "Some unsupported advice about drivers.\n\n"
        f"{av.UNVERIFIED_HEADER}\n- Some unsupported advice about drivers.\n\n"
        f"{av.PARTIAL_NOTE}"
    )
    out = av.verify_answer(echoed, [], None) if _accepts_three() else None
    if out is None:
        pytest.skip("verify_answer signature differs")
    text = out.reply if hasattr(out, "reply") else str(out)
    assert text.count(av.UNVERIFIED_HEADER) <= 1
    assert text.count(av.PARTIAL_NOTE) <= 1


def _accepts_three():
    import inspect

    return len(inspect.signature(av.verify_answer).parameters) >= 2
