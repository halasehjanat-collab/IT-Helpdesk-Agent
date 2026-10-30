"""Knowledge Base source-precedence policy.

``dataset/manifest.json`` is the single source of truth. This module reads it
and answers policy questions (eligibility, conflict winners, ``applies_to``,
department exceptions, error-code mappings). It makes no network calls and
never writes anything.

Manifest precedence rules and where they are enforced:
1. SP- documents beat KB- articles: explicit conflict winners, a ranking key,
   and a note for the model when both kinds are retrieved.
2. Only ``current`` documents are sources; a document superseded by a current
   one is excluded (``is_eligible``).
3. authoritative, then last_updated, then version (``rank_key``).
4. Department exceptions apply only for a verified department.
5. Documents beat tickets (tickets are never retrieved as guidance).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

KIT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = KIT_DIR / "dataset" / "manifest.json"

CURRENT_STATUS = "current"

_WINNER_PATTERN = re.compile(
    r"\b((?:KB|SP)-\d{3}(?:\s*(?:,|and)\s*(?:KB|SP)-\d{3})*)\s+wins?\b",
    re.IGNORECASE,
)
_DOCUMENT_ID_PATTERN = re.compile(r"(?:KB|SP)-\d{3}", re.IGNORECASE)
_ERROR_CODE_PATTERN = re.compile(r"\b[A-Za-z]{2,}-\d+\b")
_TOPIC_STOPWORDS = frozenset(
    {"the", "and", "for", "with", "you", "over", "vs", "no", "not", "of", "to"}
)


class KBPolicyError(RuntimeError):
    """The manifest is missing or malformed; callers must fail closed."""


@dataclass(frozen=True)
class Conflict:
    id: str
    topic: str
    documents: tuple[str, ...]
    winners: tuple[str, ...]
    losers: tuple[str, ...]
    terms: frozenset[str]
    resolution: str
    department_scoped: bool = False


@dataclass(frozen=True)
class KBPolicy:
    documents: dict[str, dict[str, Any]]
    superseded: frozenset[str]
    conflicts: tuple[Conflict, ...]
    department_exceptions: dict[str, dict[str, Any]]
    error_codes: dict[str, tuple[str, ...]]


def document_id(metadata: Any) -> str:
    """Normalised document ID from chunk or document metadata."""
    if not isinstance(metadata, dict):
        return ""

    return str(
        metadata.get("id") or metadata.get("document_id") or ""
    ).strip().upper()


def status_is_current(status: Any) -> bool:
    return str(status or "").strip().lower() == CURRENT_STATUS


def _as_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str) and value.strip():
        return [value.strip()]

    return []


def _topic_terms(topic: str) -> frozenset[str]:
    return frozenset(
        term
        for term in re.findall(r"[a-z0-9]+", topic.lower())
        if term not in _TOPIC_STOPWORDS and len(term) > 1
    )


def build_policy(manifest: Any) -> KBPolicy:
    """Build the policy from a parsed manifest."""
    try:
        documents = {
            str(entry["id"]).strip().upper(): entry
            for entry in manifest["documents"]
        }

        superseded: set[str] = {
            str(item["id"]).strip().upper()
            for item in manifest.get("deprecations", [])
        }

        for doc_id, entry in documents.items():
            if status_is_current(entry.get("status")):
                superseded.update(
                    item.upper() for item in _as_list(entry.get("supersedes"))
                )

        department_names = {
            str(item["department"]).strip().lower()
            for item in manifest.get("department_exceptions", [])
        }
        conflicts = []

        for raw in manifest.get("known_conflicts", []):
            members = tuple(
                str(item).strip().upper() for item in raw.get("documents", [])
            )
            resolution = str(raw.get("resolution", ""))
            match = _WINNER_PATTERN.search(resolution)
            winners = tuple(
                found.upper()
                for found in (
                    _DOCUMENT_ID_PATTERN.findall(match.group(1))
                    if match
                    else []
                )
                if found.upper() in members
            )
            # A member loses when it is not a winner and is ineligible or
            # not authoritative (a current authoritative member such as a
            # how-to that supplies the steps is not a loser).
            losers = tuple(
                member
                for member in members
                if winners
                and member not in winners
                and (
                    member not in documents
                    or not status_is_current(documents[member].get("status"))
                    or not documents[member].get("authoritative")
                )
            )
            conflicts.append(
                Conflict(
                    id=str(raw.get("id", "")),
                    topic=str(raw.get("topic", "")),
                    documents=members,
                    winners=winners,
                    losers=losers,
                    terms=_topic_terms(str(raw.get("topic", ""))),
                    resolution=resolution,
                    department_scoped=any(
                        name in resolution.lower() for name in department_names
                    ),
                )
            )

        exceptions = {
            str(item["department"]).strip().lower(): item
            for item in manifest.get("department_exceptions", [])
        }

        error_codes = {
            str(code).strip().upper(): tuple(
                item.upper() for item in _as_list(ids)
            )
            for code, ids in manifest.get("error_code_index", {}).items()
        }
    except (KeyError, TypeError, AttributeError) as exc:
        raise KBPolicyError("The Knowledge Base manifest is malformed.") from exc

    return KBPolicy(
        documents=documents,
        superseded=frozenset(superseded),
        conflicts=tuple(conflicts),
        department_exceptions=exceptions,
        error_codes=error_codes,
    )


@lru_cache(maxsize=1)
def _load_default_policy() -> KBPolicy:
    try:
        manifest = json.loads(
            DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise KBPolicyError(
            "The Knowledge Base manifest could not be read."
        ) from exc

    return build_policy(manifest)


def load_policy() -> KBPolicy:
    return _load_default_policy()


def clear_policy_cache() -> None:
    _load_default_policy.cache_clear()


def is_eligible(metadata: Any, policy: KBPolicy | None = None) -> bool:
    """True only for current, manifest-known, non-superseded documents.

    Both the document's own status and the manifest status must be current.
    Missing, unknown or unlisted documents are not eligible.
    """
    try:
        policy = policy or load_policy()
    except KBPolicyError:
        return False

    if not isinstance(metadata, dict):
        return False

    doc_id = document_id(metadata)
    entry = policy.documents.get(doc_id)

    return (
        entry is not None
        and status_is_current(metadata.get("status"))
        and status_is_current(entry.get("status"))
        and doc_id not in policy.superseded
    )


def rank_key(metadata: dict[str, Any]) -> tuple[bool, bool, str, tuple[int, ...]]:
    """Higher is better: SP-, authoritative, last_updated, then version."""
    version = tuple(
        int(part) for part in re.findall(r"\d+", str(metadata.get("version", "")))
    )

    return (
        document_id(metadata).startswith("SP-"),
        bool(metadata.get("authoritative", False)),
        str(metadata.get("last_updated", "")),
        version,
    )


def applies_to_department(
    metadata: dict[str, Any],
    department: str | None,
    policy: KBPolicy | None = None,
) -> bool:
    """Apply ``applies_to`` only for a verified department; never guess."""
    if not isinstance(department, str) or not department.strip():
        return True

    policy = policy or load_policy()
    entry = policy.documents.get(document_id(metadata), {})
    applies = _as_list(entry.get("applies_to") or metadata.get("applies_to"))

    if not applies:
        return True

    allowed = {item.lower() for item in applies}

    return "all" in allowed or department.strip().lower() in allowed


def overriding_conflicts(
    metadata: dict[str, Any],
    text: str,
    policy: KBPolicy | None = None,
) -> list[Conflict]:
    """Conflicts in which this chunk's document loses on this chunk's topic."""
    policy = policy or load_policy()
    doc_id = document_id(metadata)
    tokens = set(re.findall(r"[a-z0-9]+", str(text).lower()))
    found = []

    for conflict in policy.conflicts:
        needed = min(2, len(conflict.terms))
        if (
            doc_id in conflict.losers
            and needed
            and len(tokens & conflict.terms) >= needed
        ):
            found.append(conflict)

    return found


def error_code_documents(
    query: str,
    policy: KBPolicy | None = None,
) -> dict[str, tuple[str, ...]]:
    """Eligible documents mapped to each indexed error code in the query."""
    policy = policy or load_policy()
    mapped: dict[str, tuple[str, ...]] = {}

    for found in _ERROR_CODE_PATTERN.findall(str(query)):
        code = found.upper()
        ids = policy.error_codes.get(code)

        if not ids:
            continue

        eligible = tuple(
            doc_id
            for doc_id in ids
            if is_eligible(policy.documents.get(doc_id, {}), policy)
        )

        if eligible:
            mapped[code] = eligible

    return mapped


def verified_exception(
    department: str | None,
    policy: KBPolicy | None = None,
) -> dict[str, Any] | None:
    if not isinstance(department, str) or not department.strip():
        return None

    policy = policy or load_policy()

    return policy.department_exceptions.get(department.strip().lower())


_VPN_CONTEXT = re.compile(
    r"\b(?:vpn|nakhlaconnect|gateway|gw-\d|client)\b", re.IGNORECASE
)
_VDI_CONTEXT = re.compile(r"\b(?:vdi|workspace)\b", re.IGNORECASE)
_NEGATED = re.compile(
    r"\b(?:not|no|never|cannot|without|must not)\b|n't\b", re.IGNORECASE
)

# Instructions that a department exception forbids (KB-016, KB-002, SP-002).
# They are enforced only for a department found in the manifest exceptions.
_FINANCE_FORBIDDEN = re.compile(
    r"\b(?:switch|change|select|choose|move)\b.{0,40}\b(?:gateway|gw-\d)\b|"
    r"\b(?:re-?install|uninstall|update|upgrade|re-?configure|edit|modify|"
    r"download|install)\b.{0,50}"
    r"\b(?:client|nakhlaconnect|vpn|client\.ini|profile)\b",
    re.IGNORECASE,
)
_FINANCE_ALLOWED_ACTION = re.compile(
    r"\b(?:check|view|visit)\b.{0,30}\bstatus\s+page\b|"
    r"\b(?:security\s+operations|security\s+ticket)\b|"
    r"\b(?:p[12]|priority\s+p[12])\b",
    re.IGNORECASE,
)
_VPN_TROUBLESHOOTING_ACTION = re.compile(
    r"\b(?:approve|accept|enter|type|request|join|enrol|enroll|"
    r"reset|restart|reboot|clear|flush|connect|disconnect|"
    r"run|open|launch|reinstall|uninstall|update|upgrade|"
    r"re-?configure|edit|modify|download|install|switch|change|"
    r"select|choose|move|troubleshoot|try|use)\b",
    re.IGNORECASE,
)


def department_restriction(
    sentence: str,
    department: str | None,
    policy: KBPolicy | None = None,
    context: str = "",
) -> str | None:
    """Reason code when a sentence instructs something this department may not do.

    Only a verified department with a manifest exception is considered. Negated
    sentences ("Do not switch gateway") are statements of the restriction itself.
    """
    try:
        exception = verified_exception(department, policy)
    except KBPolicyError:
        return None

    if exception is None or _NEGATED.search(sentence):
        return None

    name = str(exception["department"]).strip().lower()
    topic_context = f"{sentence} {context}"

    if name == "finance" and _VPN_CONTEXT.search(topic_context):
        if re.search(r"security operations", sentence, re.IGNORECASE):
            return None

        if _FINANCE_FORBIDDEN.search(sentence):
            return "department_restriction:Finance"

        if (
            _VPN_TROUBLESHOOTING_ACTION.search(sentence)
            and not _FINANCE_ALLOWED_ACTION.search(sentence)
        ):
            return "department_restriction:Finance"

    if (
        name == "contractors"
        and re.search(r"\b(?:vpn|nakhlaconnect)\b", sentence, re.IGNORECASE)
        and not _VDI_CONTEXT.search(sentence)
    ):
        return "department_restriction:Contractors"

    return None


def department_guidance(
    department: str | None,
    policy: KBPolicy | None = None,
) -> tuple[str, list[str]] | None:
    """The manifest rule and its source ids for a verified department."""
    exception = verified_exception(department, policy)

    if exception is None:
        return None

    return str(exception.get("rule", "")).strip(), _as_list(
        exception.get("source")
    )


def precedence_notes(
    document_ids: Iterable[str],
    department: str | None = None,
    policy: KBPolicy | None = None,
) -> list[str]:
    """Trusted, manifest-derived notes for the model; empty when none apply."""
    try:
        policy = policy or load_policy()
    except KBPolicyError:
        return []

    ids = {str(item).strip().upper() for item in document_ids}
    notes: list[str] = []

    if any(i.startswith("SP-") for i in ids) and any(
        i.startswith("KB-") for i in ids
    ):
        notes.append(
            "Security policies (SP-) override KB articles on the same point."
        )

    exception = verified_exception(department, policy)

    for conflict in policy.conflicts:
        if conflict.winners or conflict.department_scoped:
            continue

        present = [item for item in conflict.documents if item in ids]

        if len(present) >= 2:
            notes.append(
                f"{' and '.join(present)} are documented as conflicting on "
                f"'{conflict.topic}' and the manifest gives no winner. Do not "
                "choose between them; say the answer cannot be confirmed and "
                "recommend the IT Service Desk."
            )

    if exception is not None:
        notes.append(
            f"Verified requester department: {exception['department']}. "
            f"Department exception (sources: "
            f"{', '.join(_as_list(exception.get('source')))}): "
            f"{exception.get('rule', '')} It never overrides a security policy."
        )
    elif not (isinstance(department, str) and department.strip()):
        departments = [
            item["department"]
            for item in policy.department_exceptions.values()
            if ids & {s.upper() for s in _as_list(item.get("source"))}
        ]
        if departments:
            notes.append(
                "Department-specific exceptions exist for "
                f"{', '.join(departments)}. The requester's department is "
                "not verified, so none was applied; do not guess it and say "
                "that department rules may differ."
            )

    return notes
