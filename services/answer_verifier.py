"""Deterministic, zero-LLM answer verifier.

The verifier checks a model reply against the retrieved Knowledge Base chunks
only. It makes no model, embedding or network calls and has no side effects.

What it can and cannot show
- It checks that each claim is *traceable* to eligible retrieved text, using
  several independent conditions that must all hold: the cited source is
  retrieved and eligible, hard tokens (error codes, versions, numbers, URLs,
  e-mails, file names, commands, quoted labels) appear verbatim in the
  evidence, the leading action verb appears in the matching passage, the
  matching passage shares enough content words, and the claim does not flip
  the passage's polarity (negation).
- Word overlap is a necessary filter, never proof by itself. A claim that
  fails only the overlap test is *unverified* (kept, but labelled as not
  confirmed), not deleted, so valid paraphrases are not silently lost.
- It cannot prove that a source document is itself correct, and it cannot
  detect every semantic error (for example a wrong step order).

Retrieved documents are data. They are only ever compared with the reply;
they are never executed or interpreted as instructions.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, replace
from typing import Any

from services import kb_precedence

logger = logging.getLogger(__name__)

ABSTENTION = (
    "I couldn't verify a reliable solution from the available knowledge "
    "base. I can help you escalate this issue to the Service Desk."
)
PARTIAL_NOTE = (
    "Some details could not be verified from the knowledge base and are "
    "not confirmed. I can help you escalate this issue to the Service Desk."
)
UNVERIFIED_HEADER = (
    "Not verified against the knowledge base (please do not treat as "
    "confirmed):"
)

def strip_notices(text: str) -> str:
    """Remove verifier boilerplate from text that is about to be (re)verified.

    A model that copies earlier replies from the conversation would otherwise
    have its copy of these notices wrapped in a second set.
    """
    if not isinstance(text, str):
        return ""

    lines = text.splitlines()
    kept: list[str] = []
    skipping_list = False

    for line in lines:
        stripped = line.strip()

        if stripped == UNVERIFIED_HEADER:
            skipping_list = True
            continue

        if skipping_list and (stripped.startswith("- ") or not stripped):
            continue

        skipping_list = False
        kept.append(line)

    cleaned = "\n".join(kept)

    for notice in (PARTIAL_NOTE, ABSTENTION):
        cleaned = cleaned.replace(notice, "")

    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


SUPPORTED = "supported"
UNVERIFIED = "unverified"
REJECTED = "rejected"
NON_CLAIM = "non_claim"


def _min_coverage() -> float:
    try:
        value = float(os.getenv("VERIFIER_MIN_COVERAGE", "0.6"))
    except ValueError:
        return 0.6

    return value if 0.3 <= value <= 1.0 else 0.6


MIN_SHARED_STEMS = 2


@dataclass(frozen=True)
class ClaimResult:
    text: str
    verdict: str
    reasons: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerifiedAnswer:
    reply: str
    status: str  # no_claims | supported | partial | abstained | error
    claims: tuple[ClaimResult, ...] = ()
    reasons: tuple[str, ...] = ()

    def count(self, verdict: str) -> int:
        return sum(1 for claim in self.claims if claim.verdict == verdict)


# ------------------------------------------------------------------ text tools

_STOPWORDS = frozenset("""
a an the and or but if then so to of in on at for with from by as is are was
were be been being am do does did done doing this that these those it its you
your yours we our us i me my they them their he she him her will would should
could can may might must shall have has had not no also see please use using
try first next finally there here which what when where how why who whom any
some all each other more most just only about into over under again still
once now example ok okay sure hello hi hey thanks thank help assist happy glad
sorry able need want like get got make made go goes going take taking follow
following step steps issue problem report reported seeing shows show let
know sure ensure make error code codes means mean
""".split())

_CANON = {
    "reboot": "restart", "relaunch": "restart", "login": "signin",
    "logon": "signin", "uninstall": "remove", "delete": "remove",
    "upgrade": "update", "colleague": "user", "employee": "user",
    "people": "user", "mobile": "phone",
}

_ACTION_VERBS = frozenset("""
restart reboot reinstall install uninstall update upgrade reset disable
enable delete remove clear open click select contact run connect reconnect
turn switch request submit approve share send sign log download disconnect
unlock change set
""".split())

_LEAD_WORDS = frozenset({"please", "then", "first", "next", "also", "finally"})


def _stem(word: str) -> str:
    word = word.lower()
    word = _CANON.get(word, word)

    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break

    return _CANON.get(word, word)


def _stems(text: str) -> set[str]:
    stems = set()

    for word in re.findall(r"[A-Za-z][A-Za-z']+", text):
        lowered = word.lower()

        if lowered in _STOPWORDS or len(lowered) < 3:
            continue

        stems.add(_stem(lowered))

    return stems


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


_NUMBER_WORDS = {
    "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "fifteen": "15",
}
_NUMBER_WORD = re.compile(
    r"(?<![\w@./-])(" + "|".join(_NUMBER_WORDS) + r")(?![\w@])",
    re.IGNORECASE,
)
_DIGIT_HYPHEN_WORD = re.compile(r"(?<=\d)-(?=[A-Za-z])")


def _numerals(text: str) -> str:
    """Write small number words as digits and split "6-digit" into "6 digit".

    Applied identically to replies, evidence and the user message, so "five"
    matches "5" while a different number still fails. "one" is left alone
    because it is usually a pronoun.
    """
    text = _NUMBER_WORD.sub(
        lambda match: _NUMBER_WORDS[match.group(1).lower()], str(text)
    )
    return _DIGIT_HYPHEN_WORD.sub(" ", text)


_NEGATION = re.compile(
    r"\b(?:not|no|never|cannot|without|forbidden|prohibited|disallowed|"
    r"unable)\b|n't\b",
    re.IGNORECASE,
)


def _negated(text: str) -> bool:
    return _NEGATION.search(text) is not None


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"\[(*#])")
_CLAUSE_SPLIT = re.compile(
    r"[;,]\s+|\s+(?:if|when|unless|but|because|while)\s+", re.IGNORECASE
)
_LIST_MARKER = re.compile(r"^(\s*(?:[-*\u2022]|\d+[.)])\s+)")
_BRACKET_CITE = re.compile(
    r"\s*\[\s*((?:KB|SP)-\d{3}(?:\s*[,;]\s*(?:KB|SP)-\d{3})*)\s*\]",
    re.IGNORECASE,
)
_ID = re.compile(r"\b(?:KB|SP)-\d{3}\b", re.IGNORECASE)


def _split_sentences(line: str) -> list[str]:
    return [part for part in _SENTENCE_SPLIT.split(line.strip()) if part]


def _windows(text: str) -> list[str]:
    """Sentences, adjacent sentence pairs and clauses of an evidence chunk."""
    windows: list[str] = []

    for line in str(text).splitlines():
        sentences = _split_sentences(line)
        windows.extend(sentences)
        windows.extend(
            f"{a} {b}" for a, b in zip(sentences, sentences[1:])
        )

        for sentence in sentences:
            windows.extend(
                part for part in _CLAUSE_SPLIT.split(sentence) if part
            )

    return windows


# ---------------------------------------------------------------- hard tokens

_HARD_PATTERNS = (
    ("url", re.compile(r"https?://[^\s)\]>\"']+|www\.[^\s)\]>\"']+")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("backtick", re.compile(r"`([^`]+)`")),
    ("quoted", re.compile(r"\"([^\"]{3,60})\"|\u201c([^\u201d]{3,60})\u201d")),
    ("code", re.compile(r"\b(?!KB-|SP-)[A-Z]{2,5}-\d{2,5}\b")),
    ("priority", re.compile(r"\bP[0-4]\b")),
    ("filename", re.compile(r"\b[\w-]+\.[a-z]{2,5}\b")),
    ("version", re.compile(r"\b\d+(?:\.\d+)+\b")),
    ("number", re.compile(r"\b\d+\b")),
)
_STEP_NUMBER = re.compile(r"\bstep\s+\d+\b", re.IGNORECASE)


def hard_tokens(text: str) -> list[str]:
    """Facts that must appear verbatim in evidence if the reply states them."""
    working = _STEP_NUMBER.sub(" ", text)
    working = _ID.sub(" ", working)
    tokens: list[str] = []

    for _, pattern in _HARD_PATTERNS:
        def take(match: re.Match[str]) -> str:
            groups = [g for g in match.groups() if g]
            tokens.append((groups[0] if groups else match.group(0)).strip())
            return " "

        working = pattern.sub(take, working)

    cleaned = []

    for token in tokens:
        token = token.rstrip(".,;:!?")

        if token and token.lower() not in {t.lower() for t in cleaned}:
            cleaned.append(token)

    return cleaned


def _contains_token(evidence: str, token: str) -> bool:
    wanted = _norm(token)

    if not wanted:
        return True

    pattern = r"(?<![\w.@/-])" + re.escape(wanted) + r"(?![\w@-]|\.\w)"
    return re.search(pattern, evidence) is not None


# --------------------------------------------------------------- claim classes

_UNCERTAIN = re.compile(
    r"\b(?:cannot|can't|can not|could not|couldn't|unable to|not able to)"
    r"\s+(?:\w+\s+){0,2}?(?:confirm|verify)\b|"
    r"\bnot (?:documented|covered|available in)\b|"
    r"\b(?:no|not enough) (?:documented|reliable|verified) "
    r"(?:information|evidence|guidance)\b|"
    r"\b(?:conflict|conflicting|contradict)",
    re.IGNORECASE,
)
_ESCALATE = re.compile(
    r"\b(?:contact|reach|call|ask)\b.{0,40}\b(?:service desk|it support|"
    r"helpdesk|help desk|it team)\b|\bescalat|\b(?:open|raise|file|draft|"
    r"create|prepare)\s+(?:a\s+)?(?:support\s+)?ticket\b",
    re.IGNORECASE,
)
_FALSE_ACTION = re.compile(
    r"\b(?:i|we)(?:'ve|\s+have)?\s+(?:created|opened|submitted|filed|"
    r"raised|approved|closed|updated|escalated)\b.{0,40}\bticket\b|"
    r"\bticket\b.{0,30}\b(?:has been|was|is now)\s+(?:created|opened|"
    r"submitted|filed|approved|closed)\b",
    re.IGNORECASE,
)
_CREDENTIAL_REQUEST = re.compile(
    r"\b(?:send|share|provide|tell|give|enter|type|reply with|paste|"
    r"read out)\b.{0,50}\b(?:password|passcode|pin|mfa code|otp|"
    r"one-time (?:code|password)|verification code|api key|secret|token)\b",
    re.IGNORECASE,
)


def _unresolved_conflict_notes(notes: list[str] | None) -> bool:
    return any(
        "no winner" in str(note) and "cannot be confirmed" in str(note)
        for note in notes or []
    )


_PROMPT = re.compile(r"^\s*(?:[-*\u2022]\s+|\d+[.)]\s+)?(?:[$>]\s+)?")


def _command_forms(text: str) -> frozenset[str]:
    forms: set[str] = set()

    for line in str(text).splitlines():
        whole = _norm(_PROMPT.sub("", line, count=1))

        if whole:
            forms.add(whole)

        forms.update(
            _norm(span) for span in re.findall(r"`([^`]+)`", line) if span.strip()
        )

    return frozenset(forms)


@dataclass(frozen=True)
class _Doc:
    id: str
    text_norm: str
    windows: tuple[tuple[str, frozenset[str], bool], ...]
    metadata: dict[str, Any]
    command_forms: frozenset[str] = frozenset()


@dataclass
class _Context:
    pool: dict[str, list[_Doc]]
    ineligible: set[str]
    policy: kb_precedence.KBPolicy | None
    message_norm: str
    min_coverage: float


def _build_doc(document: dict[str, Any], doc_id: str) -> _Doc:
    raw = str(document.get("text", ""))
    text = _numerals(raw)
    windows = tuple(
        (window, frozenset(_stems(window)), _negated(window))
        for window in _windows(text)
    )
    return _Doc(doc_id, _norm(text), windows, document, _command_forms(raw))


def _build_context(
    documents: list[dict[str, Any]],
    message: str,
    policy: kb_precedence.KBPolicy | None,
) -> _Context:
    pool: dict[str, list[_Doc]] = {}
    ineligible: set[str] = set()

    for document in documents or []:
        if not isinstance(document, dict):
            continue

        doc_id = kb_precedence.document_id(document)

        if not doc_id:
            continue

        if policy is None or not kb_precedence.is_eligible(document, policy):
            ineligible.add(doc_id)
            continue

        pool.setdefault(doc_id, []).append(_build_doc(document, doc_id))

    return _Context(
        pool, ineligible, policy, _norm(_numerals(message)), _min_coverage()
    )


def _evidence(ctx: _Context, ids: set[str] | None) -> list[_Doc]:
    selected = ids if ids else set(ctx.pool)
    return [d for doc_id in sorted(selected) for d in ctx.pool.get(doc_id, [])]


def _supporting_ids(
    claim_text: str,
    docs: list[_Doc],
    ctx: _Context,
    negated_override: bool | None = None,
) -> tuple[set[str], bool, bool]:
    """Return (ids with a matching passage, passage found, polarity clash).

    A passage matches when it shares at least MIN_SHARED_STEMS content words,
    reaches the minimum coverage, and contains the claim's leading action
    verb. Passages that match on words but flip polarity are reported as a
    clash only when no passage agrees with the claim's polarity.
    """
    claim_stems = _stems(claim_text)
    action = _lead_action(claim_text)
    claim_negated = (
        _negated(claim_text) if negated_override is None else negated_override
    )
    agreeing: set[str] = set()
    agreeing_cover: set[str] = set()
    opposing_cover: set[str] = set()

    if not claim_stems:
        return agreeing, False, False

    needed = min(MIN_SHARED_STEMS, len(claim_stems))

    for doc in docs:
        for _, stems, negated in doc.windows:
            shared = claim_stems & stems

            if (
                len(shared) < needed
                or len(shared) / len(claim_stems) < ctx.min_coverage
                or (action is not None and action not in stems)
            ):
                continue

            if negated == claim_negated:
                agreeing.add(doc.id)
                agreeing_cover |= shared
            else:
                opposing_cover |= shared

    # A word that only matching passages of the opposite polarity cover (for
    # example "router" in "Do not restart the router") is a conflict, even
    # when other words of the claim are supported.
    return agreeing, bool(agreeing), bool(opposing_cover - agreeing_cover)


def _lead_action(text: str) -> str | None:
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]

    while words and words[0] in _LEAD_WORDS:
        words.pop(0)

    return _stem(words[0]) if words and words[0] in _ACTION_VERBS else None


_SEGMENT_SPLIT = re.compile(
    r"\s*[;,]\s*|\s+(?:and then|and|then|but|or)\s+", re.IGNORECASE
)


_LEADING_CONJUNCTION = re.compile(
    r"^(?:and then|and|then|but|or)\s+", re.IGNORECASE
)


def _segments(text: str) -> list[str]:
    """Split a sentence into separately verifiable action clauses."""
    parts = (
        _LEADING_CONJUNCTION.sub("", part.strip(" ."))
        for part in _SEGMENT_SPLIT.split(text)
    )
    return [part for part in parts if part]


def _assess(
    segments: list[str], docs: list[_Doc], ctx: _Context
) -> tuple[set[str], bool, bool, bool]:
    """Return (supporting ids, every clause found, any clause found, clash).

    Each clause must be supported on its own, so one supported action cannot
    carry an unsupported action in the same sentence.
    """
    agreeing: set[str] = set()
    found: list[bool] = []
    clash = False
    previous_negated = False

    for segment in segments:
        # A negation carries over to following clauses that have no verb of
        # their own ("Never share your password or MFA code").
        negated = _negated(segment) or (
            previous_negated and _lead_action(segment) is None
        )
        previous_negated = negated

        if not _stems(segment):
            continue

        ids, matched, conflict = _supporting_ids(
            segment, docs, ctx, negated_override=negated
        )
        agreeing |= ids
        found.append(matched)
        clash = clash or conflict

    return agreeing, bool(found) and all(found), any(found), clash


def _lost_conflicts(doc: _Doc, text: str, ctx: _Context) -> list[Any]:
    try:
        return list(
            kb_precedence.overriding_conflicts(doc.metadata, text, ctx.policy)
        )
    except kb_precedence.KBPolicyError:
        return []


_CONDITION_LEAD = re.compile(r"^(?:if|when|unless|otherwise)\b", re.IGNORECASE)


def _instruction_clauses(analysis: str) -> list[str]:
    """Clauses of an uncertainty or escalation sentence that carry content.

    Only clauses that are themselves uncertainty/escalation wording, and bare
    "if ..." conditions without facts, are boilerplate. Anything else (an
    instruction, a field value, a number) must still be verified.
    """
    clauses: list[str] = []

    for part in re.split(r":\s+", analysis):
        clauses.extend(_segments(part))

    return [
        clause
        for clause in clauses
        if not _UNCERTAIN.search(clause)
        and not _ESCALATE.search(clause)
        and (_stems(clause) or hard_tokens(clause))
        and (
            not _CONDITION_LEAD.match(clause)
            or hard_tokens(clause)
            or _lead_action(clause)
        )
    ]


def _evaluate_shielded(
    sentence: str, analysis: str, cited: set[str], ctx: _Context
) -> ClaimResult:
    """Verify instructions hidden inside an uncertainty/escalation sentence."""
    clauses = _instruction_clauses(analysis)

    if not clauses:
        return ClaimResult(_strip_cites(sentence, set()), NON_CLAIM)

    probe = ", ".join(clauses) + "".join(f" [{i}]" for i in sorted(cited))
    result = _evaluate_sentence(probe, ctx, shielded=True)

    if result.verdict == SUPPORTED:
        return ClaimResult(
            _strip_cites(sentence, set(result.sources) & cited),
            SUPPORTED,
            (),
            result.sources,
        )

    if result.verdict == NON_CLAIM:
        return ClaimResult(_strip_cites(sentence, set()), NON_CLAIM)

    text = sentence if result.verdict == REJECTED else _strip_cites(sentence, set())
    return ClaimResult(
        text, result.verdict, (*result.reasons, "shielded_instruction")
    )


def _strip_cites(text: str, allowed: set[str]) -> str:
    def repl(match: re.Match[str]) -> str:
        kept = [
            doc_id.upper()
            for doc_id in _ID.findall(match.group(1))
            if doc_id.upper() in allowed
        ]
        return f" [{', '.join(kept)}]" if kept else ""

    return _BRACKET_CITE.sub(repl, text).strip()


def _evaluate_sentence(
    sentence: str, ctx: _Context, shielded: bool = False
) -> ClaimResult:
    cited = {doc_id.upper() for doc_id in _ID.findall(sentence)}
    numeric = _numerals(sentence)
    analysis = _ID.sub(" ", _BRACKET_CITE.sub(" ", numeric))
    stems = _stems(analysis)
    tokens = hard_tokens(numeric)
    valid_cited = cited & set(ctx.pool)
    allowed = set(ctx.pool)

    if _FALSE_ACTION.search(sentence):
        return ClaimResult(
            sentence, REJECTED, ("false_action_claim",)
        )

    is_negative = _negated(sentence)

    if _CREDENTIAL_REQUEST.search(sentence) and not is_negative:
        return ClaimResult(
            sentence, REJECTED, ("credential_request",)
        )

    if sentence.rstrip().endswith("?"):
        return ClaimResult(_strip_cites(sentence, set()), NON_CLAIM)

    if not shielded and (
        _UNCERTAIN.search(sentence) or _ESCALATE.search(sentence)
    ):
        return _evaluate_shielded(sentence, analysis, cited, ctx)

    if (
        (not stems and not tokens)
        or (sentence.rstrip().endswith(":") and not tokens
            and len(analysis.split()) <= 8)
    ):
        return ClaimResult(
            _strip_cites(sentence, allowed), NON_CLAIM
        )

    reasons: list[str] = []

    for doc_id in sorted(cited):
        manifest_entry = (
            ctx.policy.documents.get(doc_id) if ctx.policy is not None else None
        )

        if doc_id in ctx.ineligible or (
            doc_id not in ctx.pool
            and manifest_entry is not None
            and not kb_precedence.is_eligible(manifest_entry, ctx.policy)
        ):
            return ClaimResult(
                sentence, REJECTED, (f"ineligible_source:{doc_id}",)
            )

    all_chunks = _evidence(ctx, None)
    lost_by_id: dict[str, str] = {}

    for doc in all_chunks:
        lost = _lost_conflicts(doc, analysis, ctx)

        if lost:
            lost_by_id.setdefault(doc.id, lost[0].id)

    # A document that loses a conflict on this claim's topic is never evidence,
    # whether or not the sentence cites it.
    for doc_id in sorted(valid_cited):
        if doc_id in lost_by_id:
            return ClaimResult(
                sentence,
                REJECTED,
                (f"conflict_loser:{doc_id}:{lost_by_id[doc_id]}",),
            )

    usable = [d for d in all_chunks if d.id not in lost_by_id]
    barred = [d for d in all_chunks if d.id in lost_by_id]
    cited_docs = [d for d in usable if d.id in valid_cited]
    mismatch = False

    for token in tokens:
        if cited_docs and any(
            _contains_token(d.text_norm, token) for d in cited_docs
        ):
            continue

        if any(_contains_token(d.text_norm, token) for d in usable):
            if cited_docs:
                mismatch = True
            continue

        if _contains_token(ctx.message_norm, token):
            continue

        loser = next(
            (d for d in barred if _contains_token(d.text_norm, token)), None
        )

        if loser is not None:
            return ClaimResult(
                sentence,
                REJECTED,
                (f"conflict_loser:{loser.id}:{lost_by_id[loser.id]}",),
            )

        return ClaimResult(
            sentence, REJECTED, (f"invented_token:{token}",)
        )

    if not stems:
        return ClaimResult(
            _strip_cites(sentence, set()), UNVERIFIED,
            ("hard_tokens_only",),
        )

    segments = _segments(analysis)
    primary = cited_docs or usable
    agreeing, all_found, any_found, clash = _assess(segments, primary, ctx)

    if clash:
        return ClaimResult(
            sentence, REJECTED, ("polarity_conflict",)
        )

    if all_found and not mismatch:
        sources = tuple(sorted(agreeing))
        # Keep only the citations that actually support this sentence.
        text = _strip_cites(sentence, agreeing & cited)
        return ClaimResult(text, SUPPORTED, (), sources)

    if cited_docs:
        agreeing, all_found, any_found, clash = _assess(segments, usable, ctx)

        if clash:
            return ClaimResult(
                sentence, REJECTED, ("polarity_conflict",)
            )

    if not all_found and barred:
        _, barred_found, _, _ = _assess(segments, barred, ctx)

        if barred_found:
            doc_id = barred[0].id

            for doc in barred:
                if _assess(segments, [doc], ctx)[1]:
                    doc_id = doc.id
                    break

            return ClaimResult(
                sentence,
                REJECTED,
                (f"conflict_loser:{doc_id}:{lost_by_id[doc_id]}",),
            )

    if cited_docs and all_found:
        reasons.append("citation_mismatch")

    if mismatch and "citation_mismatch" not in reasons:
        reasons.append("citation_mismatch")

    if not reasons:
        if not usable:
            reasons.append("no_evidence")
        elif any_found:
            reasons.append("partially_supported")
        else:
            reasons.append("insufficient_support")

    return ClaimResult(_strip_cites(sentence, set()), UNVERIFIED, tuple(reasons))


def _evaluate_command(command: str, ctx: _Context) -> ClaimResult:
    wanted = _norm(_PROMPT.sub("", command, count=1))
    multi_word = len(wanted.split()) >= 2
    wanted_numeric = _numerals(wanted)

    for docs in ctx.pool.values():
        for doc in docs:
            # A command is matched as a whole line or backtick span, or (when
            # it has several words) as whole tokens, never as a substring of
            # ordinary words.
            if wanted and (
                wanted in doc.command_forms
                or (multi_word and _contains_token(doc.text_norm, wanted_numeric))
            ):
                return ClaimResult(command, SUPPORTED, (), ())

    return ClaimResult(command, REJECTED, ("invented_command",))


# ------------------------------------------------------------------ main entry


_HEADING = re.compile(r"^#{1,6}\s+")


def _heading_claim(
    sentence: str, ctx: _Context
) -> tuple[str | None, ClaimResult | None]:
    """Handle a Markdown heading, alone or merged with the sentence after it.

    Returns (heading line to emit or None, claim to verify or None). A short
    label with no instruction, number or negation is a non-claim. For a
    merged "## Title Sentence." the leading words are split off as the
    heading only when they are harmless (or appear verbatim in the evidence)
    and the remainder is itself fully supported, so a heading can never carry
    an unverified instruction.
    """
    marker = _HEADING.match(sentence).group(0)
    text = sentence[len(marker):].strip()
    numeric = _numerals(text)
    analysis = _ID.sub(" ", _BRACKET_CITE.sub(" ", numeric))
    words = text.split()

    if (
        len(words) <= 10
        and not re.search(r"[.!?]\s*$", text)
        and not _lead_action(analysis)
        and not hard_tokens(analysis)
        and not _negated(analysis)
    ):
        return _strip_cites(sentence, set()), None

    result = _evaluate_sentence(text, ctx)

    if result.verdict == SUPPORTED:
        return None, replace(result, text=marker + result.text)

    if len(words) < 2:
        return None, result

    for size in range(1, min(6, len(words) - 1) + 1):
        head_text = " ".join(words[:size])
        rest = " ".join(words[size:])
        head = _numerals(head_text)

        if not rest[:1].isupper() or _ID.search(head_text):
            continue

        harmless = not (
            _lead_action(head) or hard_tokens(head) or _negated(head)
        )
        in_evidence = any(
            _norm(head_text) in doc.text_norm
            for docs in ctx.pool.values()
            for doc in docs
        )

        if not (harmless or in_evidence):
            continue

        sub = _evaluate_sentence(rest, ctx)

        if sub.verdict == SUPPORTED:
            return marker + head_text, sub

    return None, result


def _abstain(claims: tuple[ClaimResult, ...], *reasons: str) -> VerifiedAnswer:
    return VerifiedAnswer(ABSTENTION, "abstained", claims, tuple(reasons))


def verify_answer(
    reply: Any,
    documents: list[dict[str, Any]] | None,
    message: str = "",
    policy_notes: list[str] | None = None,
    policy: kb_precedence.KBPolicy | None = None,
) -> VerifiedAnswer:
    """Verify a model reply against retrieved chunks. Never raises."""
    if not isinstance(reply, str) or not reply.strip():
        return VerifiedAnswer(reply if isinstance(reply, str) else "", "no_claims")

    try:
        return _verify(reply, documents or [], message, policy_notes, policy)
    except Exception:
        logger.exception("Answer verification failed; failing closed")
        return _abstain((), "verifier_error")


_NUMBERED = re.compile(r"^(\s*)(\d+)([.)])(\s+)")


def _renumber(lines: list[str]) -> list[str]:
    """Close gaps left in numbered lists after unsupported steps were removed."""
    result: list[str] = []
    expected: int | None = None

    for line in lines:
        match = _NUMBERED.match(line)

        if match:
            number = expected if expected is not None else int(match.group(2))
            line = (
                f"{match.group(1)}{number}{match.group(3)}{match.group(4)}"
                + line[match.end():]
            )
            expected = number + 1
        elif line.strip() and not line.startswith((" ", "\t")):
            expected = None

        result.append(line)

    return result


def _verify(
    reply: str,
    documents: list[dict[str, Any]],
    message: str,
    policy_notes: list[str] | None,
    policy: kb_precedence.KBPolicy | None,
) -> VerifiedAnswer:
    if policy is None:
        try:
            policy = kb_precedence.load_policy()
        except kb_precedence.KBPolicyError:
            policy = None

    original_reply = reply
    reply = strip_notices(reply)

    if not reply:
        return VerifiedAnswer(original_reply, "no_claims")

    ctx = _build_context(documents, str(message or ""), policy)
    lines = reply.splitlines()
    output: list[str] = []
    unverified: list[str] = []
    results: list[ClaimResult] = []
    changed = False
    index = 0

    while index < len(lines):
        line = lines[index]

        if line.strip().startswith("```"):
            end = index + 1
            block: list[str] = []

            while end < len(lines) and not lines[end].strip().startswith("```"):
                block.append(lines[end])
                end += 1

            kept: list[str] = []

            for command in block:
                if not command.strip():
                    kept.append(command)
                    continue

                result = _evaluate_command(command.strip(), ctx)
                results.append(result)

                if result.verdict == SUPPORTED:
                    kept.append(command)
                else:
                    changed = True

            if any(item.strip() for item in kept):
                output.extend([line, *kept, "```"])

            index = end + 1
            continue

        index += 1

        if not line.strip():
            output.append(line)
            continue

        marker_match = _LIST_MARKER.match(line)
        marker = marker_match.group(1) if marker_match else ""
        body = line[len(marker):]
        parts: list[str] = []

        for sentence in _split_sentences(body):
            heading: str | None = None

            if _HEADING.match(sentence):
                heading, result = _heading_claim(sentence, ctx)

                if heading is not None:
                    if parts:
                        output.append(marker + " ".join(parts))
                        parts, marker = [], ""

                    output.append(heading)
                    changed = changed or heading != sentence

                if result is None:
                    continue
            else:
                result = _evaluate_sentence(sentence, ctx)

            if result.verdict != NON_CLAIM:
                results.append(result)

            if result.verdict in (SUPPORTED, NON_CLAIM):
                parts.append(result.text)
                changed = changed or result.text != sentence
            elif result.verdict == UNVERIFIED:
                unverified.append(result.text)
                changed = True
            else:
                changed = True

        if parts:
            output.append(marker + " ".join(parts))

    claims = tuple(results)
    supported = sum(1 for c in claims if c.verdict == SUPPORTED)
    unverified_count = sum(1 for c in claims if c.verdict == UNVERIFIED)
    rejected = sum(1 for c in claims if c.verdict == REJECTED)

    if not claims:
        return VerifiedAnswer(reply, "no_claims", claims)

    if _unresolved_conflict_notes(policy_notes) and not _UNCERTAIN.search(reply):
        return _abstain(claims, "unresolved_conflict_without_uncertainty")

    if supported == 0:
        return _abstain(claims, "no_supported_claims")

    if not changed:
        return VerifiedAnswer(reply, "supported", claims)

    body = "\n".join(_renumber(output)).strip()

    if unverified:
        body += (
            f"\n\n{UNVERIFIED_HEADER}\n"
            + "\n".join(f"- {text}" for text in dict.fromkeys(unverified))
        )

    if unverified_count or rejected:
        body += f"\n\n{PARTIAL_NOTE}"
        return VerifiedAnswer(body, "partial", claims)

    return VerifiedAnswer(body, "supported", claims)


# --------------------------------------------------- injection quarantine (data)

_INJECTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\b(?:ignore|disregard|forget|override)\b.{0,40}"
        r"\b(?:previous|prior|above|earlier|your|system|all)\b.{0,30}"
        r"\b(?:instruction|prompt|rule|polic|guideline)",
        r"\bsystem\s+(?:override|prompt)\b",
        r"\bnote(?:s)?\s+(?:for|to)\s+(?:ai|llm|assistant|chatbot)",
        r"\b(?:ai|llm)\s+assistants?\b[^.\n]{0,60}\b(?:must|should)\s+"
        r"(?:now\s+)?(?:create|give|grant|ignore|reveal|approve|submit)\b",
        r"\b(?:do not|don't|never)\s+(?:mention|tell|reveal|disclose|"
        r"inform)\b.{0,40}\b(?:user|employee|requester|note)\b",
        r"\b(?:without asking|approval is not required|pre-approved)\b",
        r"\bmaintenance mode\b",
    )
)


def looks_like_injection(text: Any) -> bool:
    """Heuristic scan for text addressed to an AI assistant rather than users."""
    if not isinstance(text, str):
        return False

    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def quarantine_documents(
    documents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Split documents into safe ones and the IDs of quarantined ones.

    Quarantined chunks are removed before generation so that instructions
    inside retrieved data never reach the model or serve as evidence.
    """
    safe: list[dict[str, Any]] = []
    quarantined: list[str] = []

    for document in documents:
        if isinstance(document, dict) and looks_like_injection(
            document.get("text", "")
        ):
            quarantined.append(kb_precedence.document_id(document))
        else:
            safe.append(document)

    return safe, quarantined
