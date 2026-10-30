"""Guardrails AI layer for the IT Helpdesk Agent.

Uses the real `guardrails-ai` framework (`Guard` + custom `Validator`s). Three
guards exist:

* input   - runs on the user's message before anything else processes it;
* context - runs on every retrieved chunk before it can reach the model;
* output  - runs on the generated answer after the Answer Verifier.

All validators are deterministic (no model, no network, no API key). They do
not replace the existing safeguards: the Answer Verifier, KB precedence,
approval gate and tool-boundary authorization still run. Authorization is never
decided here from message text alone; the input guard only refuses requests
that are unmistakably attempts to bypass it.

A failing or crashing guard fails closed: the caller receives a fixed safe
message that never repeats the blocked content or any secret.
"""
from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any

from guardrails import Guard, OnFailAction
from guardrails.settings import settings as _guardrails_settings
from guardrails.utils.hub_telemetry_utils import HubTelemetry
from guardrails.validators import (
    FailResult,
    PassResult,
    Validator,
    register_validator,
)

from services import answer_verifier, kb_precedence

logger = logging.getLogger(__name__)

GUARD_TIMEOUT_SECONDS = float(os.getenv("GUARDRAILS_TIMEOUT_SECONDS", "5"))
MAX_INPUT_CHARS = 8000

INPUT_REFUSAL = (
    "I can't help with that request. I can't reveal credentials or internal "
    "instructions, share other people's data, or override Nakhla IT policy or "
    "the ticket approval step. I'm happy to help with a normal IT issue such "
    "as VPN, passwords, printers or devices."
)
SENSITIVE_FALLBACK = (
    "I can't share that information. Please contact the IT Service Desk if "
    "you need help."
)
UNSUPPORTED_FALLBACK = answer_verifier.ABSTENTION


# ------------------------------------------------------------- configuration

def _disable_framework_telemetry() -> None:
    """Guardrails reports anonymous usage by default; this app never sends it."""
    _guardrails_settings.rc.enable_metrics = False
    _guardrails_settings.disable_tracing = True
    HubTelemetry()._enabled = False


def _build_guard(name: str, validators: list[Validator]) -> Guard:
    # Guard.use() replaces earlier validators, so all are passed in one call.
    guard = Guard(name=name).use(*validators)
    guard.configure(allow_metrics_collection=False)
    _disable_framework_telemetry()

    for validator in getattr(guard, "_validators", []):
        validator._disable_telemetry = True

    return guard


def _flags(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


def _evidence_text(metadata: dict[str, Any], *, include_conversation: bool = True) -> str:
    documents = metadata.get("documents") or []
    parts = [str(d.get("text", "")) for d in documents if isinstance(d, dict)]

    if include_conversation:
        parts.extend(str(x) for x in metadata.get("allowed_text") or [])

    return "\n".join(parts)


# --------------------------------------------------------------- input rules

_INJECTION = tuple(
    _flags(p)
    for p in (
        r"\b(?:ignore|disregard|forget|bypass|override|skip)\b.{0,40}"
        r"\b(?:previous|prior|above|earlier|all|your|the|these|any)\b.{0,30}"
        r"\b(?:instruction|prompt|rule|polic|guideline|restriction|safeguard)",
        r"\b(?:developer|debug|dan|jailbreak|god|sudo|maintenance|unrestricted)"
        r"\s+mode\b",
        r"\byou\s+are\s+now\b.{0,40}\b(?:unrestricted|unfiltered|dan|"
        r"jailbroken|free\s+of|no\s+rules)",
        r"\bpretend\b.{0,40}\b(?:no|without)\s+(?:rules|restrictions|polic)",
        r"<\s*/?\s*(?:system|assistant)\s*>|^\s*#{1,3}\s*system\b",
    )
)
_SECRET_REQUEST = _flags(
    r"\b(?:reveal|show|print|display|dump|leak|expose|share|read|echo|"
    r"repeat|tell\s+me|what(?:'s|\s+is|\s+are))\b.{0,60}"
    r"\b(?:system\s+prompt|(?:your|the|hidden)\s+instructions?|api[\s_-]?keys?|"
    r"secrets?|access\s+tokens?|credentials?|environment\s+variables?|"
    r"\.env\b|private\s+keys?|os\.environ|(?:admin|root|service|database)\s+"
    r"passwords?)"
)
_PRIVILEGE_CLAIM = _flags(
    r"\b(?:i\s+am|i'm|this\s+is|as)\s+(?:the\s+|an?\s+|your\s+)?"
    r"(?:system\s+|it\s+|domain\s+|global\s+)?"
    r"(?:admin(?:istrator)?|root|superuser|ceo|cio|ciso|security\s+officer)\b"
    r".{0,200}\b(?:bypass|override|skip|without|no\s+need|grant|disable|"
    r"authori[sz]ed|approve|full\s+access|all\s+(?:tickets|users|records))"
)
_APPROVAL_BYPASS = _flags(
    r"\b(?:submit|create|file|open|send)\b.{0,50}\bwithout\b.{0,25}"
    r"\b(?:approv|confirm|review)|\bauto[\s-]?approve\b|"
    r"\b(?:skip|bypass)\b.{0,20}\bapproval\b"
)
_OTHER_USER_DATA = _flags(
    r"\b(?:show|list|give|get|read|dump|export|view|access|look\s*up|pull|"
    r"see)\b.{0,40}\b(?:all|every|other|another|everyone'?s?|someone(?:\s+else)?'?s?|"
    r"their|his|her)\b.{0,30}\b(?:tickets?|employees?|staff|users?|records?|"
    r"directory|salar\w*|passwords?|profiles?|personal)|"
    r"\btickets?\s+(?:of|for|from|belonging\s+to)\s+[\w.+-]+@[\w-]+\.[\w.]+|"
    r"[\w.+-]+@[\w-]+\.[\w.]+'s\s+tickets?|"
    r"\b(?:other|another|previous|different)\s+(?:users?|sessions?|employees?|"
    r"people|conversations?)\b.{0,40}\b(?:said|asked|messages?|conversation|"
    r"history|chat|told|facts)"
)


@register_validator(name="nakhla/prompt-injection", data_type="string")
class PromptInjection(Validator):
    def _validate(self, value: Any, metadata: dict[str, Any]):
        text = str(value)

        if any(p.search(text) for p in _INJECTION) or answer_verifier.looks_like_injection(
            text
        ):
            return FailResult(error_message="prompt_injection")

        return PassResult()


@register_validator(name="nakhla/secret-request", data_type="string")
class SecretRequest(Validator):
    def _validate(self, value: Any, metadata: dict[str, Any]):
        if _SECRET_REQUEST.search(str(value)):
            return FailResult(error_message="secret_request")

        return PassResult()


@register_validator(name="nakhla/privilege-claim", data_type="string")
class PrivilegeClaim(Validator):
    """An unverified "I am the administrator" claim never grants anything."""

    def _validate(self, value: Any, metadata: dict[str, Any]):
        text = str(value)

        if _PRIVILEGE_CLAIM.search(text) or _APPROVAL_BYPASS.search(text):
            return FailResult(error_message="authorization_bypass")

        return PassResult()


@register_validator(name="nakhla/other-user-data", data_type="string")
class OtherUserData(Validator):
    def _validate(self, value: Any, metadata: dict[str, Any]):
        if _OTHER_USER_DATA.search(str(value)):
            return FailResult(error_message="other_user_data")

        return PassResult()


# ------------------------------------------------------------- output rules

_SECRET_PATTERNS = tuple(
    _flags(p)
    for p in (
        r"\bsk-[A-Za-z0-9_-]{16,}",
        r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"\b(?:api[_-]?key|secret|access[_-]?token|password)\s*[:=]\s*\S{6,}",
    )
)
_SECRET_ENV_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]){2,}\d{2,4}(?![\w.])")
_TICKET_ID = re.compile(r"\bIT-\d{3,5}\b")
_CODE = re.compile(r"\b[A-Z]{2,5}-\d{3,4}\b")
_DOC_ID = re.compile(r"\[([A-Za-z]{2,3}-\d{2,4})\]")
_NON_CODE_PREFIXES = {"KB", "SP", "IT"}
_UNCERTAIN = re.compile(
    r"couldn'?t (?:verify|find|confirm)|could not (?:verify|find|confirm)|"
    r"no (?:matching|reliable|information)|not (?:found|documented|covered)|"
    r"unknown|escalat",
    re.IGNORECASE,
)


def _environment_secrets() -> list[str]:
    return [
        value
        for name, value in os.environ.items()
        if _SECRET_ENV_NAME.search(name) and len(value) >= 8
    ]


@register_validator(name="nakhla/secret-leak", data_type="string")
class SecretLeak(Validator):
    def _validate(self, value: Any, metadata: dict[str, Any]):
        text = str(value)

        if any(p.search(text) for p in _SECRET_PATTERNS) or any(
            secret in text for secret in _environment_secrets()
        ):
            return FailResult(error_message="secret_leak")

        return PassResult()


@register_validator(name="nakhla/personal-data", data_type="string")
class PersonalData(Validator):
    """Emails and phone numbers must come from evidence or the conversation."""

    def _validate(self, value: Any, metadata: dict[str, Any]):
        text = str(value)
        allowed = _evidence_text(metadata).lower()

        for email in _EMAIL.findall(text):
            if email.lower() not in allowed:
                return FailResult(error_message="personal_data")

        allowed_digits = re.sub(r"\D", "", allowed)

        for phone in _PHONE.findall(text):
            digits = re.sub(r"\D", "", phone)

            if len(digits) >= 7 and digits not in allowed_digits:
                return FailResult(error_message="personal_data")

        return PassResult()


@register_validator(name="nakhla/grounded-identifiers", data_type="string")
class GroundedIdentifiers(Validator):
    """Ticket IDs and error codes must appear in evidence; meanings need it too."""

    def _validate(self, value: Any, metadata: dict[str, Any]):
        if not metadata.get("model_answered"):
            return PassResult()

        text = str(value)
        # The user's own message is not evidence: a code they typed does not
        # give the model licence to explain it.
        evidence = _evidence_text(metadata, include_conversation=False)

        for ticket_id in _TICKET_ID.findall(text):
            if ticket_id not in evidence:
                return FailResult(error_message="invented_ticket_id")

        if _UNCERTAIN.search(text):
            return PassResult()

        for code in _CODE.findall(text):
            if code.split("-")[0] in _NON_CODE_PREFIXES:
                continue

            if code not in evidence:
                return FailResult(error_message="invented_error_code")

        return PassResult()


@register_validator(name="nakhla/citations", data_type="string")
class Citations(Validator):
    """Cited IDs must be retrieved sources; a supported answer needs a source."""

    def _validate(self, value: Any, metadata: dict[str, Any]):
        if not metadata.get("model_answered"):
            return PassResult()

        text = str(value)
        retrieved = {
            kb_precedence.document_id(d)
            for d in metadata.get("documents") or []
            if isinstance(d, dict)
        }

        for cited in _DOC_ID.findall(text):
            if cited not in retrieved:
                return FailResult(error_message="invalid_citation")

        status = metadata.get("verification_status")

        if status in {"supported", "partial"} and not metadata.get("has_citations"):
            return FailResult(error_message="missing_citation")

        return PassResult()


@register_validator(name="nakhla/evidence-and-policy", data_type="string")
class EvidenceAndPolicy(Validator):
    """Re-checks the final text against evidence, precedence and department rules."""

    def _validate(self, value: Any, metadata: dict[str, Any]):
        if not metadata.get("model_answered"):
            return PassResult()

        text = answer_verifier.strip_notices(str(value))

        if text.strip() in {"", answer_verifier.ABSTENTION}:
            return PassResult()

        documents = metadata.get("documents") or []
        department = metadata.get("department")
        policy = kb_precedence.load_policy()

        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            if kb_precedence.department_restriction(sentence, department, policy):
                return FailResult(error_message="department_policy_violation")

        result = answer_verifier.verify_answer(
            text,
            documents,
            message=str(metadata.get("message") or ""),
            policy_notes=metadata.get("policy_notes"),
            department=department,
        )

        if any(
            claim.verdict in {answer_verifier.UNVERIFIED, answer_verifier.REJECTED}
            for claim in result.claims
        ):
            return FailResult(error_message="unsupported_claim")

        return PassResult()


# ----------------------------------------------------------------- the guards

_input_guard = _build_guard(
    "nakhla-input",
    [
        PromptInjection(on_fail=OnFailAction.NOOP),
        SecretRequest(on_fail=OnFailAction.NOOP),
        PrivilegeClaim(on_fail=OnFailAction.NOOP),
        OtherUserData(on_fail=OnFailAction.NOOP),
    ],
)
_context_guard = _build_guard(
    "nakhla-context", [PromptInjection(on_fail=OnFailAction.NOOP)]
)
_output_guard = _build_guard(
    "nakhla-output",
    [
        SecretLeak(on_fail=OnFailAction.NOOP),
        PersonalData(on_fail=OnFailAction.NOOP),
        GroundedIdentifiers(on_fail=OnFailAction.NOOP),
        Citations(on_fail=OnFailAction.NOOP),
        EvidenceAndPolicy(on_fail=OnFailAction.NOOP),
    ],
)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="guardrails")


@dataclass(frozen=True)
class GuardOutcome:
    passed: bool
    text: str
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _run(guard: Guard, text: str, metadata: dict[str, Any]) -> tuple[str, ...]:
    """Reasons for every failed validator; raises on framework errors."""
    future = _executor.submit(guard.validate, text, metadata=metadata)
    outcome = future.result(timeout=GUARD_TIMEOUT_SECONDS)

    if outcome.validation_passed:
        return ()

    reasons = tuple(
        str(summary.failure_reason)
        for summary in outcome.validation_summaries
        if summary.validator_status == "fail"
    )

    return reasons or ("validation_failed",)


def _safe_run(guard: Guard, text: str, metadata: dict[str, Any]) -> tuple[str, ...]:
    try:
        return _run(guard, text, metadata)
    except FutureTimeout:
        logger.error("Guardrails validation timed out; failing closed")
        return ("guard_timeout",)
    except Exception:
        # The message is deliberately not logged: it may contain a secret.
        logger.error("Guardrails validation raised; failing closed", exc_info=False)
        return ("guard_error",)


def guard_input(message: Any) -> GuardOutcome:
    """Validate an untrusted user message. A failure yields INPUT_REFUSAL."""
    if not isinstance(message, str) or not message.strip():
        return GuardOutcome(False, INPUT_REFUSAL, ("malformed_input",))

    if len(message) > MAX_INPUT_CHARS:
        return GuardOutcome(False, INPUT_REFUSAL, ("input_too_long",))

    reasons = _safe_run(_input_guard, message, {})

    if reasons:
        logger.warning("Input guard blocked a message: %s", ", ".join(reasons))
        return GuardOutcome(False, INPUT_REFUSAL, reasons)

    return GuardOutcome(True, message)


def filter_documents(
    documents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop retrieved chunks that address the AI; return safe docs and blocked IDs."""
    safe: list[dict[str, Any]] = []
    blocked: list[str] = []

    for document in documents:
        text = document.get("text", "") if isinstance(document, dict) else ""
        reasons = _safe_run(_context_guard, str(text), {})

        if reasons:
            blocked.append(kb_precedence.document_id(document))
        else:
            safe.append(document)

    return safe, blocked


def guard_output(
    reply: Any,
    *,
    documents: list[dict[str, Any]] | None = None,
    message: str = "",
    department: str | None = None,
    policy_notes: Any = None,
    model_answered: bool = True,
    verification_status: str | None = None,
    has_citations: bool = True,
) -> GuardOutcome:
    """Validate a generated answer; a failure yields a safe fixed fallback."""
    if not isinstance(reply, str) or not reply.strip():
        return GuardOutcome(False, UNSUPPORTED_FALLBACK, ("malformed_output",))

    metadata = {
        "documents": documents or [],
        "allowed_text": [message],
        "message": message,
        "department": department,
        "policy_notes": policy_notes,
        "model_answered": model_answered,
        "verification_status": verification_status,
        "has_citations": has_citations,
    }
    reasons = _safe_run(_output_guard, reply, metadata)

    if reasons == ("invalid_citation",):
        # A fabricated marker is removed rather than the whole answer; the
        # cleaned text must still pass every check.
        retrieved = {
            kb_precedence.document_id(d) for d in documents or [] if isinstance(d, dict)
        }
        cleaned = _DOC_ID.sub(
            lambda m: m.group(0) if m.group(1) in retrieved else "", reply
        )
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
        reasons = _safe_run(_output_guard, cleaned, metadata)

        if not reasons:
            logger.warning("Output guard removed an invalid citation marker")
            return GuardOutcome(True, cleaned, ("invalid_citation_removed",))

    if not reasons:
        return GuardOutcome(True, reply)

    logger.warning("Output guard replaced an answer: %s", ", ".join(reasons))
    sensitive = {"secret_leak", "personal_data"} & set(reasons)

    return GuardOutcome(
        False,
        SENSITIVE_FALLBACK if sensitive else UNSUPPORTED_FALLBACK,
        reasons,
    )
