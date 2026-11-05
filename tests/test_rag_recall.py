"""Regression tests for knowledge-base recall (FAQ chunking, numeric ranges)."""

from services import answer_verifier, kb_ingestion, kb_precedence
from services.kb_retriever import _load_chunks


def _document(body: str) -> kb_ingestion.KnowledgeBaseDocument:
    return kb_ingestion.KnowledgeBaseDocument(
        metadata={"id": "KB-TEST"},
        body=body,
        source_filename="KB-TEST.md",
    )


def test_faq_document_is_split_one_chunk_per_entry():
    body = "# FAQ\n\n**A one?**\nAnswer one.\n\n**B two?**\nAnswer two.\n\n**C three?**\nAnswer three."
    chunks = kb_ingestion.chunk_documents([_document(body)])

    assert len(chunks) == 3
    assert "Answer one." in chunks[0].text and "Answer two." not in chunks[0].text
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_ordinary_document_keeps_window_chunking():
    body = "plain text " * 300
    chunks = kb_ingestion.chunk_documents([_document(body)], 1000, 150)

    assert len(chunks) > 1


def test_disputed_faq_entry_does_not_hide_service_desk_hours():
    texts = [
        chunk.text
        for chunk in _load_chunks()
        if kb_precedence.document_id(chunk.metadata) == "KB-017"
    ]

    assert any("08:00-18:00" in text for text in texts)


def test_time_range_end_is_not_treated_as_invented():
    evidence = "The desk is staffed 08:00-18:00 on business days."

    assert answer_verifier._contains_token(evidence, "18")
    assert not answer_verifier._contains_token(evidence, "19")
