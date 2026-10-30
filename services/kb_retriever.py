"""Semantic and lexical retrieval for the IT Helpdesk Agent."""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from services import kb_precedence
from services.kb_ingestion import (
    KnowledgeBaseChunk,
    chunk_documents,
    load_knowledge_base,
)

load_dotenv()

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "openai/text-embedding-3-small",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

DEFAULT_TOP_K = 5
CANDIDATE_MULTIPLIER = 3
MAX_POLICY_EXTRA_CHUNKS = 3

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved knowledge-base chunk and its relevance score."""

    text: str
    metadata: dict[str, Any]
    source_filename: str
    chunk_index: int
    score: float


def _stem(token: str) -> str:
    """Fold simple plurals (priorities -> priority, tickets -> ticket).

    Tokens containing digits (error codes, document IDs) are left unchanged.
    """
    if any(character.isdigit() for character in token) or len(token) <= 3:
        return token

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"

    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]

    return token


def _raw_tokens(text: str) -> list[str]:
    return [
        token
        for token in (
            raw.rstrip("._-")
            for raw in re.findall(
                r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,}",
                text.lower(),
            )
        )
        if token
    ]


def _tokenize(text: str) -> set[str]:
    """Tokenize words, document IDs, and technical error codes.

    Trailing punctuation is dropped so "priority." and "KB-015." match, and
    simple plurals are folded.
    """
    return {_stem(token) for token in _raw_tokens(text)}


_STOPWORDS = frozenset(
    "a an and any are as at be been but by can could did do does for from "
    "get got had has have how i if in into is it its me my no not of on or "
    "our should so that the their them then there these they this to up us "
    "was we were what when where which who why will with would you your "
    "am im ive about also just please help need want still again keeps "
    "tried try trying know".split()
)


def _query_terms(text: str) -> set[str]:
    """Query tokens without filler words; keeps all tokens if nothing remains."""
    terms = _tokenize(text)
    meaningful = {
        _stem(token)
        for token in _raw_tokens(text)
        if token not in _STOPWORDS
    }
    return meaningful or terms


def _term_weights(chunks: list[KnowledgeBaseChunk]) -> dict[str, float]:
    """Inverse-document-frequency weights so rare terms outrank common ones."""
    frequency: dict[str, int] = {}

    for chunk in chunks:
        for term in _tokenize(chunk.text):
            frequency[term] = frequency.get(term, 0) + 1

    total = max(len(chunks), 1)

    return {
        term: math.log(1 + total / count)
        for term, count in frequency.items()
    }


def _metadata_value(
    metadata: dict[str, Any],
    key: str,
    default: Any = "",
) -> Any:
    """Return a metadata value without assuming a specific type."""
    value = metadata.get(key, default)
    return default if value is None else value


def _document_is_eligible(
    metadata: dict[str, Any],
) -> bool:
    """Status-only check: only `current` documents are eligible."""
    return kb_precedence.status_is_current(
        _metadata_value(metadata, "status", "unknown")
    )


def _eligible_chunks(
    chunks: list[KnowledgeBaseChunk],
) -> list[KnowledgeBaseChunk]:
    """Keep chunks of current, manifest-listed, non-superseded documents."""
    policy = kb_precedence.load_policy()

    return [
        chunk
        for chunk in chunks
        if kb_precedence.is_eligible(chunk.metadata, policy)
    ]


def _lexical_score(
    query: str,
    chunk: KnowledgeBaseChunk,
    weights: dict[str, float] | None = None,
) -> float:
    """Calculate a weighted keyword-overlap score.

    Filler words never score. With `weights` (IDF over the searched chunks),
    each matching content term counts by its rarity instead of a flat 1.
    """
    query_terms = _query_terms(query)

    if not query_terms:
        return 0.0

    metadata = chunk.metadata

    title = str(
        _metadata_value(metadata, "title", "")
    )

    document_id = str(
        _metadata_value(
            metadata,
            "id",
            chunk.source_filename,
        )
    )

    tags_value = metadata.get("tags", [])

    if isinstance(tags_value, list):
        tags = " ".join(str(tag) for tag in tags_value)
    else:
        tags = str(tags_value)

    title_terms = _tokenize(title)
    id_terms = _tokenize(document_id)
    tag_terms = _tokenize(tags)
    content_terms = _tokenize(chunk.text)

    content_matches = query_terms & content_terms

    if weights is None:
        content_score = float(len(content_matches))
    else:
        content_score = sum(weights.get(term, 1.0) for term in content_matches)

    score = (
        4 * len(query_terms & title_terms)
        + 3 * len(query_terms & tag_terms)
        + 2 * len(query_terms & id_terms)
        + content_score
    )

    query_lower = query.strip().lower()

    if query_lower and query_lower in title.lower():
        score += 5

    # Give exact technical error codes extra weight.
    for term in query_terms:
        if re.fullmatch(
            r"[a-z]{2,}[-._]?\d+[a-z0-9._-]*",
            term,
        ):
            if (
                term in document_id.lower()
                or term in chunk.text.lower()
            ):
                score += 8

    return float(score)


def _openrouter_client() -> OpenAI | None:
    """Create an OpenRouter client if an API key is configured."""
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        return None

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )


def _create_embeddings(
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    """Create normalized embeddings in batches."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    client = _openrouter_client()

    if client is None:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured."
        )

    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        ordered_data = sorted(
            response.data,
            key=lambda item: item.index,
        )

        all_embeddings.extend(
            item.embedding
            for item in ordered_data
        )

    vectors = np.asarray(
        all_embeddings,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1.0

    return vectors / norms


@lru_cache(maxsize=1)
def _load_chunks() -> tuple[KnowledgeBaseChunk, ...]:
    """Load and chunk the knowledge base once per process."""
    documents = load_knowledge_base()

    chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return tuple(_eligible_chunks(chunks))


@lru_cache(maxsize=1)
def _load_chunk_embeddings() -> np.ndarray:
    """Create and cache embeddings for the eligible chunks."""
    chunks = _load_chunks()

    if not chunks:
        return np.empty((0, 0), dtype=np.float32)

    return _create_embeddings(
        [chunk.text for chunk in chunks]
    )


def _semantic_search(
    query: str,
    chunks: list[KnowledgeBaseChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Retrieve chunks using cosine similarity of normalized vectors."""
    if not chunks:
        return []

    query_vectors = _create_embeddings([query])

    chunk_vectors = _load_chunk_embeddings()

    if chunk_vectors.size == 0:
        return []

    similarities = chunk_vectors @ query_vectors[0]

    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    results: list[RetrievedChunk] = []

    for index in ranked_indices:
        chunk = chunks[int(index)]

        results.append(
            RetrievedChunk(
                text=chunk.text,
                metadata=dict(chunk.metadata),
                source_filename=chunk.source_filename,
                chunk_index=chunk.chunk_index,
                score=float(similarities[index]),
            )
        )

    return results


def _lexical_search(
    query: str,
    chunks: list[KnowledgeBaseChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Retrieve chunks using weighted keyword overlap."""
    scored: list[
        tuple[float, KnowledgeBaseChunk]
    ] = []

    weights = _term_weights(chunks)

    for chunk in chunks:
        score = _lexical_score(query, chunk, weights)

        if score > 0:
            scored.append((score, chunk))

    scored.sort(
        key=lambda item: (
            item[0],
            *kb_precedence.rank_key(item[1].metadata),
        ),
        reverse=True,
    )

    maximum_score = max(
        (score for score, _ in scored),
        default=0.0,
    )

    results: list[RetrievedChunk] = []

    for score, chunk in scored[:top_k]:
        normalized_score = (
            score / maximum_score
            if maximum_score
            else 0.0
        )

        results.append(
            RetrievedChunk(
                text=chunk.text,
                metadata=dict(chunk.metadata),
                source_filename=chunk.source_filename,
                chunk_index=chunk.chunk_index,
                score=float(normalized_score),
            )
        )

    return results


def _pool_candidates(
    pool: list[KnowledgeBaseChunk],
    doc_id: str,
    department: str | None,
    policy: kb_precedence.KBPolicy,
) -> list[KnowledgeBaseChunk]:
    """Chunks of one document that the policy would allow to be returned."""
    return [
        chunk
        for chunk in pool
        if kb_precedence.document_id(chunk.metadata) == doc_id
        and kb_precedence.applies_to_department(
            chunk.metadata, department, policy
        )
        and not kb_precedence.overriding_conflicts(
            chunk.metadata, chunk.text, policy
        )
    ]


def _pick_chunk(
    candidates: list[KnowledgeBaseChunk],
    query: str,
    needles: list[str],
) -> KnowledgeBaseChunk | None:
    """Prefer a chunk containing a needle, else the best keyword match."""
    for needle in needles:
        for chunk in candidates:
            if needle.lower() in chunk.text.lower():
                return chunk

    scored = [
        (_lexical_score(query, chunk), chunk) for chunk in candidates
    ]
    scored = [item for item in scored if item[0] > 0]

    return max(scored, key=lambda item: item[0])[1] if scored else None


def _apply_policy(
    query: str,
    results: list[RetrievedChunk],
    pool: list[KnowledgeBaseChunk],
    top_k: int,
    department: str | None,
) -> list[RetrievedChunk]:
    """Apply the manifest precedence policy to candidate results.

    Runs after both semantic and lexical search, so ranking alone never
    decides a conflict: lower-precedence chunks on a conflict topic are
    removed, and policy-required chunks (conflict winners, error-code
    documents, the verified department's exception) are pinned. Pinned
    chunks are marked with metadata["policy_reason"] and score 0.0.
    """
    policy = kb_precedence.load_policy()
    kept: list[RetrievedChunk] = []
    winners_needed: dict[str, str] = {}

    for result in results:
        metadata = result.metadata

        if not kb_precedence.is_eligible(metadata, policy):
            continue

        if not kb_precedence.applies_to_department(
            metadata, department, policy
        ):
            continue

        lost = kb_precedence.overriding_conflicts(
            metadata, result.text, policy
        )

        if lost:
            query_terms = _tokenize(query)

            for conflict in lost:
                if not (query_terms & conflict.terms):
                    continue

                for winner in conflict.winners:
                    winners_needed.setdefault(winner, conflict.id)
            continue

        kept.append(result)

    kept = kept[:top_k]
    seen = {(item.source_filename, item.chunk_index) for item in kept}
    present = {kb_precedence.document_id(item.metadata) for item in kept}
    extras: list[RetrievedChunk] = []

    def pin(chunk: KnowledgeBaseChunk | None, reason: str) -> None:
        if chunk is None or len(extras) >= MAX_POLICY_EXTRA_CHUNKS:
            return

        key = (chunk.source_filename, chunk.chunk_index)

        if key in seen:
            return

        seen.add(key)
        present.add(kb_precedence.document_id(chunk.metadata))
        extras.append(
            RetrievedChunk(
                text=chunk.text,
                metadata={**chunk.metadata, "policy_reason": reason},
                source_filename=chunk.source_filename,
                chunk_index=chunk.chunk_index,
                score=0.0,
            )
        )

    for winner, conflict_id in winners_needed.items():
        if winner not in present:
            pin(
                _pick_chunk(
                    _pool_candidates(pool, winner, department, policy),
                    query,
                    [],
                ),
                f"conflict_winner:{conflict_id}",
            )

    for code, doc_ids in kb_precedence.error_code_documents(
        query, policy
    ).items():
        for doc_id in doc_ids:
            if doc_id not in present:
                pin(
                    _pick_chunk(
                        _pool_candidates(pool, doc_id, department, policy),
                        query,
                        [code],
                    ),
                    f"error_code:{code}",
                )

    exception = kb_precedence.verified_exception(department, policy)

    if exception is not None and exception.get("source"):
        name = str(exception["department"])
        # Department metadata can describe unrelated subjects (for example,
        # Engineering's software exception must not enter a VPN answer).
        # Pin only when the query overlaps the exception's actual rule/topic.
        query_terms = _query_terms(query)
        rule_terms = _query_terms(str(exception.get("rule", "")))
        relevant = bool(query_terms & rule_terms)

        if name.lower() == "finance" and re.search(
            r"\b(?:vpn|nakhlaconnect|gateway|gw-\d|client|remote\s+access)\b",
            query,
            re.IGNORECASE,
        ):
            relevant = True
        elif name.lower() == "contractors" and re.search(
            r"\b(?:vpn|nakhlaconnect|gateway|gw-\d|client|remote\s+work|vdi)\b",
            query,
            re.IGNORECASE,
        ):
            relevant = True
        elif name.lower() == "engineering" and re.search(
            r"\b(?:ubuntu|linux|admin|elevat|software\s+center|install|"
            r"software|developer)\b", query, re.IGNORECASE
        ):
            relevant = True

        if relevant:
            for source in exception["source"]:
                source_id = str(source).upper()

                if source_id not in present:
                    pin(
                        _pick_chunk(
                            _pool_candidates(pool, source_id, department, policy),
                            query,
                            [f"**{name}**", name],
                        ),
                        f"department_exception:{name}",
                    )
                    if source_id in present:
                        break

    return kept + extras


def retrieve_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    department: str | None = None,
) -> list[RetrievedChunk]:
    """Retrieve relevant knowledge chunks for a user query.

    Semantic search is preferred when embeddings are available.
    Lexical search is used when the embedding service is unavailable.
    The manifest precedence policy is applied to both.

    `department` must come from a verified employee-directory result; when
    it is None no department-specific filtering or exception is applied.
    """
    if not isinstance(query, str) or not query.strip():
        return []

    if top_k <= 0:
        return []

    try:
        chunks = list(_load_chunks())
        kb_precedence.load_policy()
    except kb_precedence.KBPolicyError:
        logger.error("Knowledge Base policy unavailable; returning nothing.")
        return []

    if not chunks:
        return []

    if not _tokenize(query):
        return []

    candidate_k = min(len(chunks), top_k * CANDIDATE_MULTIPLIER)

    try:
        # Avoid returning unrelated results for a low-similarity query.
        results = [
            result
            for result in _semantic_search(
                query.strip(),
                chunks,
                candidate_k,
            )
            if result.score > 0
        ]

    except Exception:
        # Do not expose provider errors or credentials.
        # Continue to provide keyword retrieval if embeddings fail.
        results = _lexical_search(
            query.strip(),
            chunks,
            candidate_k,
        )

    return _apply_policy(
        query.strip(),
        results,
        chunks,
        top_k,
        department,
    )


def clear_retrieval_cache() -> None:
    """Clear in-memory document and embedding caches."""
    _load_chunk_embeddings.cache_clear()
    _load_chunks.cache_clear()
