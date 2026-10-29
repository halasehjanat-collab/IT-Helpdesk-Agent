import asyncio
from types import SimpleNamespace

import pytest

import agent as agent_module
from services.kb_retriever import RetrievedChunk


def chunk(doc_id, status="current", title=None, filename=None):
    return RetrievedChunk(
        text=f"If the connection drops, restart the client and sign in again. ({doc_id})",
        metadata={
            "id": doc_id,
            "title": title if title is not None else f"Title {doc_id}",
            "status": status,
            "filename": filename if filename is not None else f"{doc_id.lower()}.md",
        },
        source_filename=f"{doc_id.lower()}.md",
        chunk_index=0,
        score=0.9,
    )


class FakeOpenAI:
    """Stands in for the OpenAI client class; reply text or an error is chosen per test."""

    reply: str | Exception | None = ""

    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @classmethod
    def _create(cls, **kwargs):
        if isinstance(cls.reply, Exception):
            raise cls.reply
        message = SimpleNamespace(content=cls.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


@pytest.fixture
def run(monkeypatch):
    """Run the real run_agent with fixed retrieval and a fake model client (no network)."""

    def _run(chunks, reply, api_key="test-key"):
        monkeypatch.setattr(agent_module, "retrieve_chunks", lambda query, k: chunks)
        monkeypatch.setattr(agent_module, "OpenAI", FakeOpenAI)
        if api_key:
            monkeypatch.setenv("OPENROUTER_API_KEY", api_key)
        else:
            monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        FakeOpenAI.reply = reply
        return asyncio.run(agent_module.run_agent("How do I fix NX-781?"))

    return _run


def ids(result):
    return [c["id"] for c in result["citations"]]


def test_only_explicitly_cited_retrieved_ids_are_returned(run):
    result = run([chunk("KB-002"), chunk("KB-003"), chunk("KB-004")], "Restart the client [KB-002].")
    assert ids(result) == ["KB-002"]
    assert result["citations"][0] == {"id": "KB-002", "title": "Title KB-002", "filename": "kb-002.md"}


def test_multiple_chunks_of_one_source_give_one_citation(run):
    result = run([chunk("KB-002"), chunk("KB-002")], "See [KB-002] and again KB-002.")
    assert ids(result) == ["KB-002"]


def test_reply_without_citation_ids_returns_none(run):
    assert run([chunk("KB-002")], "Restart the VPN client.")["citations"] == []


def test_cited_id_that_was_not_retrieved_is_rejected(run):
    result = run([chunk("KB-002")], "Do this [KB-002]. Also see [KB-009] and [SP-001].")
    assert ids(result) == ["KB-002"]
    assert run([chunk("KB-002")], "See [KB-999].")["citations"] == []


@pytest.mark.parametrize("status", ["deprecated", "draft", "legacy", "superseded"])
def test_non_current_documents_cannot_be_cited(run, status):
    result = run([chunk("KB-001", status=status), chunk("KB-002")], "Use [KB-001] and [KB-002].")
    assert ids(result) == ["KB-002"]


def test_documents_with_invalid_metadata_are_not_cited(run):
    result = run([chunk("KB-002", title=" "), chunk("KB-003", filename=" "), chunk("KB-004")],
                 "[KB-002] [KB-003] [KB-004]")
    assert ids(result) == ["KB-004"]


def test_model_failure_returns_no_citations(run):
    result = run([chunk("KB-002")], RuntimeError("provider down"))
    assert result["citations"] == []
    assert "could not connect" in result["reply"]
    assert result["usage"] is None


def test_missing_api_key_returns_no_citations(run):
    result = run([chunk("KB-002")], "ignored", api_key=None)
    assert result["citations"] == []
    assert "not configured" in result["reply"]


@pytest.mark.parametrize("empty", ["", "   \n", None])
def test_empty_model_response_returns_no_citations(run, empty):
    result = run([chunk("KB-002")], empty)
    assert result["citations"] == []
    assert "could not generate a response" in result["reply"]


def test_model_echoing_fallback_ids_in_failure_text_is_not_cited(run):
    # The failure fallback names the retrieved sources; those mentions are not evidence.
    result = run([chunk("KB-002")], RuntimeError("x"))
    assert "KB-002" in result["reply"] and result["citations"] == []


@pytest.mark.parametrize("reply", [
    "I cannot confirm the solution from the available knowledge base [KB-003]. Contact the IT Service Desk.",
    "I could not confirm this from the sources. See [KB-003].",
    "Unable to verify the fix [KB-003].",
])
def test_unconfirmed_answers_do_not_cite_unrelated_sources(run, reply):
    assert run([chunk("KB-003")], reply)["citations"] == []


def test_response_contract_is_unchanged(run):
    result = run([chunk("KB-002")], "Do this [KB-002].")
    assert set(result) == {"answer", "reply", "citations", "tool_calls", "pending_action", "usage", "state"}
    assert result["answer"] == result["reply"]
    assert result["tool_calls"] == [] and result["pending_action"] is None and result["state"] == "completed"
    assert agent_module.generate_reply("q", [], []) is not None  # 2-tuple wrapper still exists


def test_generate_reply_keeps_two_tuple_signature(monkeypatch):
    monkeypatch.setattr(agent_module, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    FakeOpenAI.reply = "Hello [KB-002]"
    reply, usage = agent_module.generate_reply("q", [], [{"id": "KB-002"}])
    assert reply == "Hello [KB-002]" and usage is None


def test_chat_endpoint_returns_filtered_citations(client, fake_mcp, run, monkeypatch):
    monkeypatch.setattr(agent_module, "retrieve_chunks", lambda q, k: [chunk("KB-002"), chunk("KB-003")])
    monkeypatch.setattr(agent_module, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    FakeOpenAI.reply = "Restart the client [KB-002]."
    body = client.post("/api/chat", json={"session_id": "c1", "message": "VPN NX-781 help"}).json()
    assert [c["id"] for c in body["citations"]] == ["KB-002"]
    assert body["pending_action"] is None and body["tool_calls"] == [] and "reply" in body
    fake_mcp.create_approved_ticket.assert_not_called()
