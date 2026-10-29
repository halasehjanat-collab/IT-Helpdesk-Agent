"""Read-only, deterministic ticket intelligence for the IT Helpdesk Agent.

Provides conservative duplicate-candidate detection, observed-count trend
summaries and draft knowledge-improvement proposals.

Guarantees:
- No LLM calls, no embedding calls and no network access.
- Nothing is written: tickets are never merged, closed or modified, and
  Knowledge Base content is never published.
- Ticket text is untrusted data. It is only tokenised and compared; it is
  never interpreted as an instruction.
- Results are evidence for a human reviewer, not decisions.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from services.helpdesk_ticket_db import TicketDatabase, TicketDatabaseError
from services.kb_retriever import _document_is_eligible

KIT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE_PATH = KIT_DIR / "dataset" / "tickets" / "historical_tickets.json"
DEFAULT_MANIFEST_PATH = KIT_DIR / "dataset" / "manifest.json"

DEFAULT_SIMILARITY_THRESHOLD = 0.6
DEFAULT_WINDOW_DAYS = 30
DEFAULT_MAX_CANDIDATES = 5
DEFAULT_MIN_TREND_SAMPLE = 20
DEFAULT_MIN_PROPOSAL_TICKETS = 2
RESOLUTION_EXCERPT_CHARS = 400

RESOLVED_STATUSES = {"resolved", "closed"}
REQUIRED_TICKET_FIELDS = ("ticket_id", "subject", "description", "category", "created_at")

# Error codes look like NX-781 or NID-401. Document and ticket IDs are not error codes.
_ERROR_CODE_PATTERN = re.compile(r"\b([A-Za-z]{2,5}-\d{2,5})\b")
_NON_ERROR_CODE_PREFIXES = ("KB-", "SP-", "IT-")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have i in is it its my not of on or so that the "
    "this to was we when with after again still cannot cant can will would".split()
)


class IntelligenceDataError(RuntimeError):
    """The data needed for analysis could not be read safely."""


# --------------------------------------------------------------------------- helpers
def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def extract_error_codes(*texts: Any) -> set[str]:
    """Normalised (upper-case) error codes found in the given texts."""
    codes: set[str] = set()
    for text in texts:
        if not isinstance(text, str):
            continue
        for match in _ERROR_CODE_PATTERN.findall(text):
            code = match.upper()
            if not code.startswith(_NON_ERROR_CODE_PREFIXES):
                codes.add(code)
    return codes


def normalize_tokens(*texts: Any) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        if isinstance(text, str):
            tokens.update(
                token for token in _TOKEN_PATTERN.findall(text.lower())
                if token not in _STOPWORDS and len(token) > 1
            )
    return tokens


def text_similarity(left: set[str], right: set[str]) -> float:
    """Jaccard similarity of two token sets (a signal only, never proof)."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _is_valid_ticket(ticket: Any) -> bool:
    if not isinstance(ticket, dict):
        return False
    for field in REQUIRED_TICKET_FIELDS:
        if not isinstance(ticket.get(field), str) or not ticket[field].strip():
            return False
    return _parse_timestamp(ticket["created_at"]) is not None


def _valid_tickets(tickets: Iterable[Any]) -> tuple[list[dict[str, Any]], int]:
    valid: list[dict[str, Any]] = []
    skipped = 0
    for ticket in tickets or []:
        if _is_valid_ticket(ticket):
            valid.append(ticket)
        else:
            skipped += 1
    return valid, skipped


# --------------------------------------------------------------------------- loaders (read-only)
def load_archive(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Read the historical JSON archive without modifying it."""
    archive_path = Path(path) if path else DEFAULT_ARCHIVE_PATH
    try:
        payload = json.loads(archive_path.read_text(encoding="utf-8"))
        tickets = payload["tickets"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise IntelligenceDataError("The historical ticket archive could not be read.") from exc
    if not isinstance(tickets, list):
        raise IntelligenceDataError("The historical ticket archive has an unexpected shape.")
    return tickets


def load_database_tickets(db_path: Path | str | None) -> list[dict[str, Any]]:
    """Read tickets from an existing ticket database; never creates or initialises one."""
    if db_path is None or not Path(db_path).is_file():
        return []
    try:
        return TicketDatabase(db_path).list_tickets()
    except TicketDatabaseError as exc:
        raise IntelligenceDataError("The ticket database could not be read.") from exc


def load_ticket_dataset(
    archive_path: Path | str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Archive tickets plus database-backed tickets, with the origin counts."""
    archive = load_archive(archive_path)
    database = load_database_tickets(db_path)
    return {
        "tickets": archive + database,
        "sources": {"historical_archive": len(archive), "ticket_database": len(database)},
    }


class KnowledgeCatalog:
    """KB document IDs with status and eligibility taken from the manifest."""

    def __init__(self, documents: dict[str, dict[str, Any]], error_code_index: dict[str, list[str]]):
        self.documents = documents
        self.error_code_index = error_code_index

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> "KnowledgeCatalog":
        try:
            entries = manifest["documents"]
            index = manifest.get("error_code_index", {})
            deprecations = {d["id"]: d for d in manifest.get("deprecations", [])}
            current_ids = {d["id"] for d in entries if str(d.get("status", "")).lower() == "current"}
            superseded = {
                str(target)
                for d in entries if d.get("id") in current_ids
                for target in (d.get("supersedes") if isinstance(d.get("supersedes"), list) else [d.get("supersedes")])
                if target
            }
            documents = {}
            for entry in entries:
                doc_id = str(entry["id"]).upper()
                status = str(entry.get("status", "unknown")).lower()
                reason = None
                if not _document_is_eligible({"status": status}):
                    reason = f"status is {status}"
                elif doc_id in superseded or doc_id in deprecations:
                    reason = "superseded by a current document"
                documents[doc_id] = {
                    "title": str(entry.get("title", "")),
                    "status": status,
                    "eligible": reason is None,
                    "ineligible_reason": reason,
                    "superseded_by": (deprecations.get(doc_id) or {}).get("superseded_by"),
                }
        except (KeyError, TypeError, AttributeError) as exc:
            raise IntelligenceDataError("The Knowledge Base manifest is malformed.") from exc
        return cls(documents, {str(k).upper(): list(v) for k, v in index.items()})

    @classmethod
    def load(cls, path: Path | str | None = None) -> "KnowledgeCatalog":
        manifest_path = Path(path) if path else DEFAULT_MANIFEST_PATH
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise IntelligenceDataError("The Knowledge Base manifest could not be read.") from exc
        return cls.from_manifest(manifest)

    def classify_refs(self, refs: Any) -> dict[str, list[Any]]:
        """Split KB references into eligible, ineligible and unknown (never guessed)."""
        result: dict[str, list[Any]] = {"eligible": [], "ineligible": [], "unknown": []}
        for ref in refs if isinstance(refs, list) else []:
            ref_id = str(ref).strip().upper()
            document = self.documents.get(ref_id)
            if document is None:
                result["unknown"].append(ref_id)
            elif document["eligible"]:
                if ref_id not in result["eligible"]:
                    result["eligible"].append(ref_id)
            else:
                result["ineligible"].append(
                    {"id": ref_id, "reason": document["ineligible_reason"], "superseded_by": document["superseded_by"]}
                )
        return result


# --------------------------------------------------------------------------- duplicates
def find_duplicate_candidates(
    target: dict[str, Any],
    tickets: Iterable[Any],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    window_days: int | None = DEFAULT_WINDOW_DAYS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> dict[str, Any]:
    """Return possible duplicates of `target` with the evidence for each.

    A candidate needs the same category and either a shared normalised error
    code or text similarity at or above the threshold. Two tickets that both
    name error codes but share none are never candidates. A candidate is only
    "strong" when it shares an error code and the same requester.
    Nothing is merged, closed or modified.
    """
    target_codes = extract_error_codes(target.get("subject"), target.get("description"))
    target_tokens = normalize_tokens(target.get("subject"), target.get("description"))
    target_time = _parse_timestamp(target.get("created_at"))
    target_requester = str(target.get("requester_email", "")).strip().lower()
    target_category = target.get("category")
    target_id = target.get("ticket_id")

    valid, skipped = _valid_tickets(tickets)
    candidates: list[dict[str, Any]] = []

    for ticket in valid:
        if target_id and ticket["ticket_id"] == target_id:
            continue
        if not target_category or ticket["category"] != target_category:
            continue

        days_apart = None
        if target_time is not None:
            days_apart = abs((_parse_timestamp(ticket["created_at"]) - target_time).total_seconds()) / 86400
            if window_days is not None and days_apart > window_days:
                continue

        codes = extract_error_codes(ticket["subject"], ticket["description"])
        shared_codes = sorted(target_codes & codes)
        if target_codes and codes and not shared_codes:
            continue  # conflicting error codes: strong evidence these are different incidents

        similarity = round(text_similarity(target_tokens, normalize_tokens(ticket["subject"], ticket["description"])), 3)
        if not shared_codes and similarity < similarity_threshold:
            continue

        same_requester = bool(target_requester) and target_requester == str(ticket.get("requester_email", "")).strip().lower()
        reasons = ["same category"]
        if shared_codes:
            reasons.append("shared error code: " + ", ".join(shared_codes))
        if similarity >= similarity_threshold:
            reasons.append(f"text similarity {similarity} >= threshold {similarity_threshold}")
        if same_requester:
            reasons.append("same requester")
        if days_apart is not None:
            reasons.append(f"created {days_apart:.1f} days apart")

        candidates.append({
            "ticket_id": ticket["ticket_id"],
            "status": ticket.get("status"),
            "level": "strong" if shared_codes and same_requester else "possible",
            "reasons": reasons,
            "evidence": {
                "shared_error_codes": shared_codes,
                "text_similarity": similarity,
                "same_requester": same_requester,
                "same_category": True,
                "days_apart": None if days_apart is None else round(days_apart, 2),
            },
        })

    candidates.sort(key=lambda c: (c["level"] != "strong", -bool(c["evidence"]["shared_error_codes"]),
                                   -c["evidence"]["text_similarity"], c["ticket_id"]))
    return {
        "candidates": candidates[:max(0, max_candidates)],
        "tickets_considered": len(valid),
        "skipped_malformed": skipped,
        "similarity_threshold": similarity_threshold,
        "window_days": window_days,
        "automatic_action": "none",
        "note": "Candidates are evidence for human review; similarity is a signal, not proof of duplication.",
    }


# --------------------------------------------------------------------------- trends
def _period_key(moment: datetime, period: str) -> str:
    if period == "week":
        year, week, _ = moment.isocalendar()
        return f"{year}-W{week:02d}"
    return f"{moment.year}-{moment.month:02d}"


def summarize_trends(
    tickets: Iterable[Any],
    *,
    period: str = "month",
    group_by: str = "category",
    start: str | None = None,
    end: str | None = None,
    min_sample: int = DEFAULT_MIN_TREND_SAMPLE,
    source: str = "unspecified",
) -> dict[str, Any]:
    """Observed ticket counts per period and group. Never predicts or claims significance."""
    if period not in {"week", "month"}:
        raise ValueError("period must be 'week' or 'month'")
    if group_by not in {"category", "department", "status", "priority", "error_code"}:
        raise ValueError("unsupported group_by")

    start_time, end_time = _parse_timestamp(start), _parse_timestamp(end)
    valid, skipped = _valid_tickets(tickets)

    buckets: dict[str, Counter] = defaultdict(Counter)
    sample = 0
    first = last = None
    for ticket in valid:
        created = _parse_timestamp(ticket["created_at"])
        if (start_time and created < start_time) or (end_time and created > end_time):
            continue
        if group_by == "error_code":
            groups = sorted(extract_error_codes(ticket["subject"], ticket["description"])) or ["(no error code)"]
        else:
            groups = [str(ticket.get(group_by) or "(unspecified)")]
        for group in groups:
            buckets[_period_key(created, period)][group] += 1
        sample += 1
        first = created if first is None or created < first else first
        last = created if last is None or created > last else last

    periods = sorted(buckets)
    limitations = []
    if sample < min_sample:
        limitations.append(f"sample size {sample} is below the minimum of {min_sample}")
    if len(periods) < 2:
        limitations.append("fewer than two periods with data; no comparison is possible")

    observed_changes = []
    if len(periods) >= 2:
        previous, current = periods[-2], periods[-1]
        for group in sorted(set(buckets[previous]) | set(buckets[current])):
            before, after = buckets[previous][group], buckets[current][group]
            observed_changes.append({
                "group": group, "previous_period": previous, "current_period": current,
                "previous_count": before, "current_count": after, "difference": after - before,
            })
        limitations.append("the latest period may be incomplete")

    return {
        "data_source": source,
        "time_range": {"first_ticket": first.isoformat() if first else None, "last_ticket": last.isoformat() if last else None},
        "period": period,
        "group_by": group_by,
        "sample_size": sample,
        "skipped_malformed": skipped,
        "counts_by_period": {key: dict(sorted(buckets[key].items())) for key in periods},
        "observed_changes": observed_changes if sample >= min_sample else [],
        "insufficient_data": bool(limitations) and (sample < min_sample or len(periods) < 2),
        "limitations": limitations,
        "statistical_significance": "not assessed",
        "predictions": "none; observed counts only",
    }


# --------------------------------------------------------------------------- knowledge proposals
def propose_knowledge_updates(
    tickets: Iterable[Any],
    catalog: KnowledgeCatalog,
    *,
    min_tickets: int = DEFAULT_MIN_PROPOSAL_TICKETS,
) -> dict[str, Any]:
    """Draft, unverified KB improvement proposals grouped by error code.

    Only resolved or closed tickets that carry a resolution are used. A
    resolution is never assumed to be correct, so every proposal is
    unverified, not publishable and requires authorised human review.
    KB references come only from the ticket `kb_refs` field and are checked
    against the manifest eligibility rules.
    """
    valid, skipped = _valid_tickets(tickets)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded = Counter()

    for ticket in valid:
        resolution = ticket.get("resolution")
        if str(ticket.get("status", "")).lower() not in RESOLVED_STATUSES:
            excluded["not_resolved"] += 1
        elif not isinstance(resolution, str) or not resolution.strip():
            excluded["missing_resolution"] += 1
        else:
            codes = extract_error_codes(ticket["subject"], ticket["description"])
            if not codes:
                excluded["no_error_code"] += 1
            for code in codes:
                groups[code].append(ticket)

    proposals = []
    for code in sorted(groups):
        members = sorted(groups[code], key=lambda t: t["ticket_id"])
        evidence, uncertainty = [], []
        ref_status: dict[str, dict[str, Any]] = {"eligible": {}, "ineligible": {}, "unknown": {}}
        for ticket in members:
            classified = catalog.classify_refs(ticket.get("kb_refs"))
            for ref in classified["eligible"]:
                ref_status["eligible"][ref] = catalog.documents[ref]["title"]
            for ref in classified["ineligible"]:
                ref_status["ineligible"][ref["id"]] = ref
            for ref in classified["unknown"]:
                ref_status["unknown"][ref] = True
            evidence.append({
                "ticket_id": ticket["ticket_id"],
                "status": ticket.get("status"),
                "resolved_at": ticket.get("resolved_at"),
                "kb_refs": [str(r) for r in ticket.get("kb_refs") or []],
                "resolution_excerpt": ticket["resolution"].strip()[:RESOLUTION_EXCERPT_CHARS],
            })

        resolution_tokens = [normalize_tokens(t["resolution"]) for t in members]
        if any(text_similarity(a, b) < 0.15 for i, a in enumerate(resolution_tokens) for b in resolution_tokens[i + 1:]):
            uncertainty.append("resolutions differ substantially; they may conflict")
        if len(members) < min_tickets:
            uncertainty.append(f"only {len(members)} supporting ticket(s); minimum is {min_tickets}")
        if ref_status["ineligible"]:
            uncertainty.append("tickets reference ineligible KB documents: " + ", ".join(sorted(ref_status["ineligible"])))
        if ref_status["unknown"]:
            uncertainty.append("tickets reference unknown KB IDs: " + ", ".join(sorted(ref_status["unknown"])))

        indexed = [doc for doc in catalog.error_code_index.get(code, []) if str(doc).upper() in catalog.documents]
        eligible_indexed = [str(d).upper() for d in indexed if catalog.documents[str(d).upper()]["eligible"]]
        if not indexed:
            kind = "possible_kb_gap"
            uncertainty.append("this error code is not in the manifest error-code index")
        elif not eligible_indexed:
            kind = "update_ineligible_coverage"
        else:
            kind = "review_existing_article"

        proposals.append({
            "proposal_id": "KP-" + hashlib.sha256(
                (code + "|" + ",".join(e["ticket_id"] for e in evidence)).encode()).hexdigest()[:12],
            "error_code": code,
            "kind": kind,
            "source_ticket_ids": [e["ticket_id"] for e in evidence],
            "evidence": evidence,
            "kb_references": {
                "eligible": ref_status["eligible"],
                "ineligible": list(ref_status["ineligible"].values()),
                "unknown": sorted(ref_status["unknown"]),
            },
            "eligible_kb_for_error_code": eligible_indexed,
            "verified": False,
            "verification_note": "Resolutions come from ticket text and are not independently verified.",
            "insufficient_evidence": len(members) < min_tickets,
            "uncertainty": uncertainty,
            "publishable": False,
            "requires_human_review": True,
        })

    return {
        "proposals": proposals,
        "tickets_considered": len(valid),
        "skipped_malformed": skipped,
        "excluded": dict(excluded),
        "automatic_action": "none; nothing is published or modified",
    }
