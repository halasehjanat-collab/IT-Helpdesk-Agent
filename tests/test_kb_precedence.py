"""Offline tests for the Knowledge Base source-precedence policy.

No network, no live model, no database. Embeddings are faked; the real
manifest and real KB files are only read from disk.
"""

import asyncio
import copy
import json
from unittest.mock import AsyncMock

import numpy as np
import pytest

import agent as agent_module
import app as app_module
from services import kb_precedence as kp
from services import kb_retriever as kr
from services.kb_ingestion import KnowledgeBaseChunk

POLICY = kp.load_policy()
MANIFEST = json.loads(kp.DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
EXPECTED_ELIGIBLE = {
    "SP-001", "SP-002", "KB-002", "KB-003", "KB-004", "KB-006", "KB-007",
    "KB-008", "KB-009", "KB-010", "KB-011", "KB-012", "KB-013", "KB-014",
    "KB-015", "KB-016", "KB-017",
}
EXPECTED_EXCLUDED = {"KB-001", "KB-005", "KB-018"}


def meta(doc_id, **override):
    entry = POLICY.documents[doc_id]
    base = {
        key: entry[key]
        for key in (
            "id", "title", "status", "authoritative", "version",
            "last_updated", "applies_to",
        )
    }
    base.update(override)
    return base


def make_chunk(doc_id, text, index=0, **override):
    return KnowledgeBaseChunk(
        text=text,
        metadata=meta(doc_id, **override),
        source_filename=f"{doc_id}.md",
        chunk_index=index,
    )


def ids_of(results):
    return [kp.document_id(item.metadata) for item in results]


@pytest.fixture
def search(monkeypatch):
    """Run retrieve_chunks over a synthetic pool with lexical or fake-embedding search."""

    def _search(chunks, query, mode="lexical", department=None, top_k=5, favour=None):
        monkeypatch.setattr(kr, "_load_chunks", lambda: tuple(chunks))

        if mode == "lexical":
            def boom(*args, **kwargs):
                raise RuntimeError("embeddings unavailable")

            monkeypatch.setattr(kr, "_create_embeddings", boom)
        else:
            count = len(chunks)
            query_vector = np.full(count, 0.05, dtype=np.float32)
            for index, weight in (favour or {}).items():
                query_vector[index] = weight
            query_vector /= np.linalg.norm(query_vector)
            monkeypatch.setattr(
                kr, "_create_embeddings", lambda texts, batch_size=64: query_vector[None, :]
            )
            monkeypatch.setattr(
                kr, "_load_chunk_embeddings", lambda: np.eye(count, dtype=np.float32)
            )

        return kr.retrieve_chunks(query, top_k, department=department)

    return _search


# ---------------------------------------------------------------- eligibility


def test_real_manifest_eligibility_matches_inspected_statuses():
    eligible = {
        doc_id for doc_id, entry in POLICY.documents.items() if kp.is_eligible(entry)
    }
    assert eligible == EXPECTED_ELIGIBLE
    assert set(POLICY.documents) - eligible == EXPECTED_EXCLUDED


@pytest.mark.parametrize("status", [None, "", "unknown", "review", "legacy", "superseded"])
def test_missing_or_unknown_status_is_not_eligible(status):
    assert not kp.is_eligible(meta("KB-003", status=status))


def test_status_is_normalised_and_unlisted_documents_are_rejected():
    assert kp.is_eligible(meta("KB-003", status=" Current "))
    assert not kp.is_eligible({**meta("KB-003"), "id": "KB-999"})
    assert not kp.is_eligible({"status": "current"})
    assert not kp.is_eligible("KB-003")


@pytest.mark.parametrize("doc_id", sorted(EXPECTED_EXCLUDED))
def test_front_matter_cannot_override_the_manifest_status(doc_id):
    assert not kp.is_eligible(meta(doc_id, status="current"))


def test_document_superseded_by_a_current_document_is_not_eligible():
    manifest = copy.deepcopy(MANIFEST)
    entries = {d["id"]: d for d in manifest["documents"]}
    entries["KB-003"]["supersedes"] = "KB-004"
    policy = kp.build_policy(manifest)
    assert not kp.is_eligible(meta("KB-004"), policy)

    entries["KB-003"]["status"] = "draft"
    assert kp.is_eligible(meta("KB-004"), kp.build_policy(manifest))


def test_malformed_manifest_raises_and_policy_fails_closed(monkeypatch, search):
    with pytest.raises(kp.KBPolicyError):
        kp.build_policy({"nope": []})

    def unavailable():
        raise kp.KBPolicyError("down")

    monkeypatch.setattr(kp, "load_policy", unavailable)
    assert not kp.is_eligible(meta("KB-003"))
    assert kp.precedence_notes(["KB-003"]) == []
    assert search([make_chunk("KB-003", "office wifi guide")], "wifi") == []


# ------------------------------------------------------------------ conflicts


def test_conflicts_c1_to_c6_are_parsed_from_the_manifest():
    found = {c.id: c for c in POLICY.conflicts}
    assert set(found) == {"C1", "C2", "C3", "C4", "C5", "C6"}
    assert found["C1"].winners == ("KB-002",)
    assert set(found["C1"].losers) == {"KB-001", "KB-018"}
    assert set(found["C2"].winners) == {"SP-001", "KB-004"}
    assert found["C2"].losers == ("KB-005",)
    assert found["C3"].winners == ("KB-015",) and found["C3"].losers == ("KB-017",)
    assert found["C4"].winners == ("SP-002",)
    assert found["C4"].losers == ("KB-017",)  # KB-007 supplies the steps and is not a loser
    assert found["C5"].winners == () and found["C5"].department_scoped
    assert found["C6"].winners == ("KB-015",) and found["C6"].losers == ()


@pytest.mark.parametrize("mode", ["lexical", "semantic"])
def test_c3_non_authoritative_kb017_cannot_override_authoritative_kb015(search, mode):
    chunks = [
        make_chunk("KB-017", "Major incident P1 threshold: 5 or more colleagues."),
        make_chunk("KB-015", "Major incident P1 threshold: more than 3 users."),
        make_chunk("KB-003", "Office wifi guide."),
    ]
    results = search(
        chunks, "What is the major incident P1 threshold?", mode, favour={0: 1.0, 1: 0.5}
    )
    assert "KB-017" not in ids_of(results)
    assert "KB-015" in ids_of(results)


def test_c3_winner_is_pinned_when_search_did_not_return_it(search):
    chunks = [
        make_chunk("KB-017", "Major incident P1 threshold: 5 or more colleagues.", 0),
        make_chunk("KB-003", "Office wifi guide one.", 1),
        make_chunk("KB-006", "Account lockout guide two.", 2),
        make_chunk("KB-015", "Major incident P1 threshold: more than 3 users.", 3),
    ]
    results = search(
        chunks,
        "major incident P1 threshold",
        "semantic",
        top_k=1,
        favour={0: 1.0, 1: 0.9, 2: 0.8, 3: 0.01},
    )
    pinned = [r for r in results if r.metadata.get("policy_reason")]
    assert ids_of(results).count("KB-017") == 0
    assert [kp.document_id(r.metadata) for r in pinned] == ["KB-015"]
    assert pinned[0].metadata["policy_reason"] == "conflict_winner:C3"
    assert pinned[0].score == 0.0


@pytest.mark.parametrize("mode", ["lexical", "semantic"])
def test_c4_security_policy_beats_kb017_sms_advice(search, mode):
    chunks = [
        make_chunk("KB-017", "No phone with you? Ask the desk for a one-time SMS code."),
        make_chunk("SP-002", "SMS and voice codes are not allowed for MFA."),
        make_chunk("KB-007", "Use a backup code when you have no phone."),
    ]
    results = search(
        chunks, "I have no phone, can the desk send an SMS code?", mode, favour={0: 1.0}
    )
    assert "KB-017" not in ids_of(results)
    assert {"SP-002", "KB-007"} <= set(ids_of(results))


@pytest.mark.parametrize("mode", ["lexical", "semantic"])
def test_c1_deprecated_and_draft_vpn_sources_are_never_returned(search, mode):
    chunks = [
        make_chunk("KB-001", "Minimum VPN client version is 4.8.", status="current"),
        make_chunk("KB-018", "Edit client.ini to skip the VPN client version check.", status="current"),
        make_chunk("KB-002", "Minimum VPN client version is 5.4."),
    ]
    results = search(chunks, "minimum VPN client version", mode, favour={0: 1.0, 1: 1.0})
    assert ids_of(results) == ["KB-002"]


@pytest.mark.parametrize("mode", ["lexical", "semantic"])
def test_c2_phone_password_reset_legacy_source_is_never_returned(search, mode):
    chunks = [
        make_chunk("KB-005", "The desk reads a password reset over the phone.", status="current"),
        make_chunk("KB-004", "Reset your password with self-service only."),
        make_chunk("SP-001", "The desk must not reset a password over the phone."),
    ]
    results = search(chunks, "password reset over the phone", mode, favour={0: 1.0})
    assert "KB-005" not in ids_of(results)
    assert {"KB-004", "SP-001"} <= set(ids_of(results))


def test_c5_c6_do_not_drop_documents_and_c6_never_retrieves_tickets(search):
    found = {c.id: c for c in POLICY.conflicts}
    assert found["C6"].losers == ()
    results = search([make_chunk("KB-015", "Priority P1 ticket rule.")], "priority rule ticket")
    assert ids_of(results) == ["KB-015"]


def test_non_conflict_content_of_kb017_is_still_retrievable(search):
    results = search(
        [make_chunk("KB-017", "Printer queue FAQ: clear the print queue.")], "printer queue"
    )
    assert ids_of(results) == ["KB-017"]


def test_unresolved_conflict_is_reported_as_uncertainty_not_a_winner():
    manifest = copy.deepcopy(MANIFEST)
    manifest["known_conflicts"] = [
        {"id": "X1", "topic": "Backup window", "documents": ["KB-003", "KB-006"],
         "resolution": "Needs review by the owners."}
    ]
    policy = kp.build_policy(manifest)
    assert policy.conflicts[0].winners == () and policy.conflicts[0].losers == ()
    notes = kp.precedence_notes(["KB-003", "KB-006"], None, policy)
    assert any("no winner" in n and "cannot be confirmed" in n for n in notes)
    assert not any("no winner" in n for n in kp.precedence_notes(["KB-003"], None, policy))


def test_real_manifest_has_no_unresolved_conflict_notes():
    notes = kp.precedence_notes(["KB-002", "KB-016", "KB-015", "KB-017"], "Finance")
    assert not any("no winner" in n for n in notes)


# ------------------------------------------------------- SP, authority, ranking


def test_security_policy_ranks_above_a_newer_authoritative_kb_article():
    sp = meta("SP-001")
    kb = meta("KB-002", last_updated="2030-01-01", version="9.0")
    assert kp.rank_key(sp) > kp.rank_key(kb)


def test_rank_key_orders_authority_then_date_then_version():
    base = meta("KB-003", authoritative=False, last_updated="2026-01-01", version="1.0")
    assert kp.rank_key({**base, "authoritative": True}) > kp.rank_key(base)
    assert kp.rank_key({**base, "last_updated": "2026-02-01"}) > kp.rank_key(base)
    assert kp.rank_key({**base, "version": "1.10"}) > kp.rank_key({**base, "version": "1.9"})
    newer_but_not_authoritative = {**base, "last_updated": "2030-01-01"}
    assert kp.rank_key({**base, "authoritative": True}) > kp.rank_key(newer_but_not_authoritative)


def test_lexical_tie_is_broken_by_authority(search):
    chunks = [
        make_chunk("KB-017", "Shared tie wording."),
        make_chunk("KB-003", "Shared tie wording."),
    ]
    assert ids_of(search(chunks, "shared tie wording")) == ["KB-003", "KB-017"]


def test_note_states_that_security_policy_overrides_kb():
    assert "Security policies (SP-) override KB articles on the same point." in (
        kp.precedence_notes(["SP-002", "KB-007"])
    )
    assert not any("Security policies" in n for n in kp.precedence_notes(["KB-007"]))


# ------------------------------------------------- applies_to and departments


@pytest.mark.parametrize("department, kept", [
    ("Contractors", False), ("contractors", False), ("Finance", True),
    ("Engineering", True), (None, True), ("  ", True),
])
def test_applies_to_is_only_applied_for_a_verified_department(search, department, kept):
    chunks = [make_chunk("KB-002", "VPN client guide."), make_chunk("KB-003", "VPN wifi guide.")]
    results = search(chunks, "VPN guide", department=department)
    assert ("KB-002" in ids_of(results)) is kept
    assert "KB-003" in ids_of(results)  # applies_to: [all]


def test_verified_finance_department_gets_its_exception_chunk_and_rule(search):
    chunks = [
        make_chunk("KB-002", "VPN client troubleshooting steps.", 0),
        make_chunk("KB-016", "Department rules header only.", 1),
        make_chunk("KB-016", "| **Finance** | profile | straight to Security Operations |", 2),
    ]
    results = search(chunks, "VPN troubleshooting", department="Finance")
    pinned = [r for r in results if r.metadata.get("policy_reason")]
    assert [r.metadata["policy_reason"] for r in pinned] == ["department_exception:Finance"]
    assert "**Finance**" in pinned[0].text
    notes = kp.precedence_notes(ids_of(results), "Finance")
    assert any("Verified requester department: Finance" in n and "Security Operations" in n for n in notes)


def test_contractor_gets_vdi_exception_instead_of_vpn_article(search):
    chunks = [
        make_chunk("KB-002", "VPN client troubleshooting steps.", 0),
        make_chunk("KB-016", "| **Contractors** | No VPN. Only the VDI portal |", 1),
    ]
    results = search(chunks, "VPN troubleshooting", department="Contractors")
    assert "KB-002" not in ids_of(results)
    assert "KB-016" in ids_of(results)


def test_unverified_department_applies_no_exception_and_says_so(search):
    chunks = [
        make_chunk("KB-002", "VPN client troubleshooting steps.", 0),
        make_chunk("KB-016", "| **Finance** | Security Operations |", 1),
    ]
    results = search(chunks, "VPN troubleshooting")
    assert not any(r.metadata.get("policy_reason") for r in results)
    notes = kp.precedence_notes(ids_of(results), None)
    assert any("not verified" in n and "do not guess" in n for n in notes)
    assert not any("Verified requester department" in n for n in notes)


def test_department_without_a_manifest_exception_adds_nothing():
    assert kp.verified_exception("Marketing") is None
    assert kp.precedence_notes(["KB-002"], "Marketing") == []


# ------------------------------------------------------------------ error codes


def test_error_code_lookup_returns_only_eligible_documents():
    assert kp.error_code_documents("I get nx-781 and NX-403") == {
        "NX-781": ("KB-002",),
        "NX-403": ("KB-002",),
    }
    assert kp.error_code_documents("ZZ-999 unknown, plain words") == {}


def test_error_code_mapped_only_to_ineligible_documents_is_ignored():
    manifest = copy.deepcopy(MANIFEST)
    manifest["error_code_index"]["OLD-100"] = ["KB-001"]
    assert kp.error_code_documents("OLD-100", kp.build_policy(manifest)) == {}


@pytest.mark.parametrize("mode", ["lexical", "semantic"])
def test_error_code_pins_the_mapped_document_even_if_search_misses_it(search, mode):
    fillers = [make_chunk("KB-003", f"my laptop shows laptop wifi filler {i}.", i) for i in range(6)]
    chunks = fillers + [
        make_chunk("KB-002", "NX-781 client version too old.", 6),
        make_chunk("KB-018", "NX-781 fix: edit client.ini.", 7, status="current"),
        make_chunk("KB-001", "NX-781 legacy note.", 8, status="current"),
    ]
    results = search(
        chunks, "my laptop shows nx-781", mode, top_k=1, favour={0: 1.0, 1: 0.9, 2: 0.8}
    )
    pinned = [r for r in results if r.metadata.get("policy_reason")]
    if mode == "semantic":
        assert [kp.document_id(r.metadata) for r in pinned] == ["KB-002"]
        assert pinned[0].metadata["policy_reason"] == "error_code:NX-781"
    assert "KB-002" in ids_of(results)
    assert not {"KB-018", "KB-001"} & set(ids_of(results))


def test_unknown_error_code_pins_nothing(search):
    chunks = [make_chunk("KB-003", "Office wifi guide ZZ-999.")]
    assert not any(r.metadata.get("policy_reason") for r in search(chunks, "ZZ-999"))


def test_pinned_chunks_are_capped():
    assert kr.MAX_POLICY_EXTRA_CHUNKS == 3


# --------------------------------------------- real KB: KB-018 never retrieved


@pytest.mark.parametrize("query", [
    "edit client.ini to skip the VPN version check",
    "NX-781 VPN client version",
    "minimum NakhlaConnect client 4.8",
    "reset a password over the phone",
    "VPN community wiki tips",
])
def test_real_kb_never_retrieves_deprecated_or_draft_documents(monkeypatch, query):
    def boom(*args, **kwargs):
        raise RuntimeError("embeddings unavailable")

    kr.clear_retrieval_cache()
    monkeypatch.setattr(kr, "_create_embeddings", boom)
    try:
        for department in (None, "Finance", "Contractors", "Engineering"):
            results = kr.retrieve_chunks(query, 5, department=department)
            assert results
            assert not EXPECTED_EXCLUDED & set(ids_of(results))
    finally:
        kr.clear_retrieval_cache()


def test_real_kb_error_code_query_returns_the_mapped_current_document(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("embeddings unavailable")

    kr.clear_retrieval_cache()
    monkeypatch.setattr(kr, "_create_embeddings", boom)
    try:
        assert "KB-002" in ids_of(kr.retrieve_chunks("NX-781", 5))
    finally:
        kr.clear_retrieval_cache()


def test_retrieve_chunks_signature_and_result_shape_are_backward_compatible(search):
    result = search([make_chunk("KB-003", "Office wifi guide.")], "wifi guide")[0]
    assert (result.text, result.source_filename, result.chunk_index) == (
        "Office wifi guide.", "KB-003.md", 0
    )
    assert isinstance(result.metadata, dict) and isinstance(result.score, float)
    assert kr.retrieve_chunks("   ") == [] and kr.retrieve_chunks("q", 0) == []


# ------------------------------------------------------ agent citation policy


def doc(doc_id, **override):
    return {**meta(doc_id), "filename": f"{doc_id}.md", "text": "x", **override}


@pytest.mark.parametrize("doc_id", sorted(EXPECTED_EXCLUDED))
def test_citations_reject_excluded_documents_even_with_spoofed_current_status(doc_id):
    documents = [doc(doc_id, status="current"), doc("KB-002")]
    cited = agent_module.select_citations(f"See [{doc_id}] and [KB-002].", documents, True)
    assert [c["id"] for c in cited] == ["KB-002"]


def test_citations_still_require_retrieval_and_an_explicit_reference():
    documents = [doc("KB-002"), doc("KB-003")]
    cited = agent_module.select_citations("Do this [KB-002]. Also [KB-004].", documents, True)
    assert [c["id"] for c in cited] == ["KB-002"]
    assert agent_module.select_citations("[KB-002]", documents, False) == []


def run_agent_with(monkeypatch, department=None):
    seen = {}

    def fake_retrieve(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return [kr.RetrievedChunk(
            text="If the VPN connection drops, restart the client and sign in again.",
            metadata=meta("KB-002"), source_filename="KB-002.md",
            chunk_index=0, score=0.9)]

    def fake_generate(message, history, documents, **kwargs):
        seen["notes"] = kwargs.get("policy_notes")
        return "Restart the client [KB-002].", None, True

    monkeypatch.setattr(agent_module, "retrieve_chunks", fake_retrieve)
    monkeypatch.setattr(agent_module, "generate_reply_with_status", fake_generate)
    kwargs = {} if department is None else {"department": department}
    result = asyncio.run(agent_module.run_agent("VPN problem", **kwargs))
    return result, seen


def test_run_agent_without_department_keeps_the_original_retrieval_call(monkeypatch):
    result, seen = run_agent_with(monkeypatch)
    assert seen["args"] == ("VPN problem", 5) and seen["kwargs"] == {}
    assert any("not verified" in n for n in seen["notes"])
    assert set(result) >= {"reply", "answer", "citations", "tool_calls", "pending_action", "usage", "state"}
    assert [c["id"] for c in result["citations"]] == ["KB-002"]


def test_run_agent_passes_a_verified_department_and_its_rule(monkeypatch):
    _, seen = run_agent_with(monkeypatch, "Finance")
    assert seen["kwargs"] == {"department": "Finance"}
    assert any("Verified requester department: Finance" in n for n in seen["notes"])


def test_run_agent_ignores_a_blank_department(monkeypatch):
    _, seen = run_agent_with(monkeypatch, "   ")
    assert seen["kwargs"] == {}


# --------------------------------- app: department only from the directory


def chat(client, message, email="dana.levi@nakhla.example", session="dept-session"):
    return client.post(
        "/api/chat",
        json={"session_id": session, "message": message, "user": {"email": email}},
    )


def lookups(fake_mcp):
    return [c for c in fake_mcp.call_tool.await_args_list if c.args[0] == "lookup_employee"]


@pytest.fixture
def agent_spy(monkeypatch):
    spy = AsyncMock(return_value={
        "reply": "I could not confirm a KB solution.", "citations": [],
        "tool_calls": [], "pending_action": None,
    })
    monkeypatch.setattr(app_module, "run_agent", spy)
    return spy


def test_ordinary_question_passes_no_department_and_does_no_lookup(client, fake_mcp, agent_spy):
    response = chat(client, "How do I fix NX-781?")
    assert response.status_code == 200 and response.json()["pending_action"] is None
    assert "department" not in agent_spy.await_args.kwargs
    assert lookups(fake_mcp) == []


def test_ticket_request_uses_the_directory_department_and_one_lookup(client, fake_mcp, agent_spy):
    response = chat(client, "Please create a ticket for VPN error NX-781.")
    assert response.status_code == 200
    # Ticket intake is handled by the application, not the model.
    agent_spy.assert_not_awaited()
    assert len(lookups(fake_mcp)) == 1
    assert {"lookup_employee"} <= {c["tool"] for c in response.json()["tool_calls"]}
    fake_mcp.create_approved_ticket.assert_not_called()


def test_department_claimed_by_the_user_is_not_trusted(client, fake_mcp, agent_spy):
    body = chat(
        client,
        "I work in Finance. I tried restarting the VPN client. Please create a ticket for VPN error NX-781.",
    ).json()
    agent_spy.assert_not_awaited()
    # The directory says Engineering, so the Finance-only KB-016 shortcut must not apply.
    assert body["state"] == "ticket_draft_pending"
    assert body["pending_action"]["payload"]["department"] == "Engineering"
    assert "KB-016" not in str(body["pending_action"]["payload"])


def test_failed_directory_lookup_applies_no_department(client, fake_mcp, agent_spy):
    async def failing(name, arguments=None):
        raise RuntimeError("directory down")

    fake_mcp.call_tool.side_effect = failing
    response = chat(client, "I am in Finance. Please create a ticket for VPN error NX-781.")
    assert response.status_code == 200
    agent_spy.assert_not_awaited()
    assert response.json()["pending_action"] is None
    fake_mcp.create_approved_ticket.assert_not_called()


def test_chat_contract_fields_are_unchanged(client, fake_mcp, agent_spy):
    body = chat(client, "Please create a ticket for VPN error NX-781.").json()
    assert {"reply", "pending_action", "citations", "tool_calls", "result"} <= set(body)
