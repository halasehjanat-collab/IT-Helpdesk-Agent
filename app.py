import asyncio
import hmac
import json
import logging
import os
import re
import sys
import uuid
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jsonschema import Draft202012Validator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Tool
from pydantic import BaseModel, ConfigDict, Field

from agent import is_memory_chatter, run_agent
from services import chat_memory, guardrails_service, kb_precedence, ticket_intelligence
from services.approval_store import ApprovalStoreError, SQLiteApprovalStore
from services.local_ticket_store import (
    SQLiteTicketMetadataStore,
    TicketStoreError,
    configured_database_path,
    payload_digest,
)

READ_ONLY_MCP_TOOL_NAMES = frozenset(
    {
        "lookup_employee",
        "get_service_status",
        "search_tickets",
        "get_ticket",
    }
)

TEST_DRAFT_TRIGGER = "/test-draft"

TICKET_CREATION_PATTERNS = (
    re.compile(
        r"\b(?:create|open|raise|submit|file|log|make)\s+"
        r"(?:a\s+)?(?:support\s+)?ticket\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:prepare|draft)\s+(?:a\s+)?(?:support\s+)?ticket\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bprepare\s+it\s+for\s+me\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:need|want|would\s+like)\s+(?:to\s+)?"
        r"(?:(?:open|create|raise|submit|file|log)\s+)?"
        r"(?:a\s+)?(?:support\s+)?ticket\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bescalate(?:\s+(?:this|it|my\s+\w+|the\s+\w+|\w+))?\b",
        re.IGNORECASE,
    ),
)

_TICKET_FILLER = re.compile(
    r"\b(?:please|pls|kindly|can|could|would|will|you|i|me|my|us|it|this|"
    r"that|the|a|an|to|for|now|just|also|then|and|ok|okay|yes|yeah|sure|"
    r"help|with|support|ticket|tickets|create|open|raise|submit|file|log|"
    r"make|prepare|draft|want|need|like|issue|problem|about|on|of|is|do|"
    r"did|all|thanks|thank|escalate|service|desk)\b",
    re.IGNORECASE,
)
_AFFIRMATION = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|please|go\s+ahead|do\s+it|proceed)"
    r"(?:[\s,.!]+(?:please|help\s+me|do\s+it|go\s+ahead|proceed|prepare\s+it|"
    r"create\s+it|sounds\s+good|thanks))*[\s.!]*$",
    re.IGNORECASE,
)
_ESCALATION_QUESTION = re.compile(
    r"escalate|support\s+ticket\s+for\s+your\s+approval", re.IGNORECASE
)
_TICKET_INTAKE_HANDOFF = re.compile(
    r"^\s*(?:how\s+(?:long|much|many|often|far|do|does|can)|what(?:'s|\s+is|\s+are)|"
    r"when|where|why|who|is\s+there|are\s+there)\b[^\n]{0,200}\?\s*$|"
    r"\bi only want\b|\bonly asking\b|"
    r"\b(?:don't|do not|never)\s+(?:create|prepare|open|raise|submit|file|log)"
    r"\s+(?:a\s+)?(?:support\s+)?ticket\b|"
    r"\b(?:cancel|stop)\s+(?:the\s+)?(?:ticket|draft|intake)\b|"
    r"\bwhat should i (?:do|try)\b|"
    r"\bhow can i (?:fix|resolve|troubleshoot)\b|"
    r"\bhow do i (?:fix|resolve|troubleshoot)\b|"
    r"\bwhich(?:\s+one)?\s+should i follow\b|"
    r"\bwhich (?:policy|guidance|instruction) should i follow\b|"
    r"\bwhat does\b.{0,100}\bmean\b|"
    r"\bcan you (?:check|explain|help me understand)\b",
    re.IGNORECASE,
)


def _leaves_ticket_intake(message: str) -> bool:
    """Recognize a new chat question or explicit cancellation during intake."""
    return bool(_TICKET_INTAKE_HANDOFF.search(_unquoted_user_text(message)))


def _asks_to_explain_quoted_ticket_instruction(message: str) -> bool:
    quoted_request = re.search(
        r"""["“‘].{0,160}\b(?:create|open|raise|submit|file|log)\s+"""
        r"""(?:a\s+)?(?:support\s+)?ticket\b.{0,160}["”’]""",
        message,
        re.IGNORECASE | re.DOTALL,
    )
    asks_for_meaning = re.search(
        r"\b(?:explain|what does it mean|what it means|meaning)\b",
        _unquoted_user_text(message),
        re.IGNORECASE,
    )
    return bool(quoted_request and asks_for_meaning)


def _is_bare_ticket_request(message: str) -> bool:
    """True when the message names no issue of its own ("prepare it for me")."""
    text = _unquoted_user_text(message)

    if _FOLLOWED_GUIDANCE.search(text):
        return True

    # "Escalate" and short mistypes such as "escalate date" continue the
    # current issue; they are not a new issue description.
    if re.fullmatch(r"\s*escalate(?:\s+\w+)?[.!?\s]*", text, re.IGNORECASE):
        has_issue_marker = bool(
            re.search(r"\b[A-Z]{2,5}-?\d{3,4}\b", text, re.IGNORECASE)
            or _ticket_service(text)
        )
        if not has_issue_marker:
            return True

    remainder = re.sub(r"[^\w\s]", " ", _TICKET_FILLER.sub(" ", text))
    return not remainder.split()


def _requests_ticket_creation(message: str) -> bool:
    """Detect explicit ticket requests outside quoted or pasted text."""
    if not isinstance(message, str):
        return False

    normalized = _unquoted_user_text(message)

    for pattern in TICKET_CREATION_PATTERNS:
        for match in pattern.finditer(normalized):
            preceding_text = normalized[
                max(
                    normalized.rfind(".", 0, match.start()),
                    normalized.rfind("!", 0, match.start()),
                    normalized.rfind("?", 0, match.start()),
                    normalized.rfind("\n", 0, match.start()),
                ) + 1:match.start()
            ]

            if re.search(
                r"\b(?:don't|do not|never|not|shouldn't|should not|"
                r"won't|wouldn't)\b(?:\s+\w+){0,5}\s*$",
                preceding_text,
                re.IGNORECASE,
            ):
                continue

            if re.match(
                r"\s*(?:how|what|when|where|why|do|does|"
                r"can i|should i|is there|can you tell me)\b",
                preceding_text,
                re.IGNORECASE,
            ):
                continue

            return True

    return False

logger = logging.getLogger(__name__)

if os.getenv("HELPDESK_PIPELINE_DIAGNOSTICS") == "1":
    logger.setLevel(logging.INFO)

TICKET_SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "dataset"
    / "schemas"
    / "ticket.schema.json"
)

with TICKET_SCHEMA_PATH.open(encoding="utf-8") as schema_file:
    TICKET_SCHEMA = json.load(schema_file)

Draft202012Validator.check_schema(TICKET_SCHEMA)
TICKET_SCHEMA_VALIDATOR = Draft202012Validator(TICKET_SCHEMA)


@dataclass(frozen=True)
class ApprovedTicketAuthorization:
    action_id: str
    payload: dict[str, Any]


class ReadOnlyMCPClient:
    def __init__(self, session: ClientSession, tools: tuple[Tool, ...]):
        self._session = session
        self.tools = tools
        self._tool_names = {tool.name for tool in tools}

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        if name not in self._tool_names:
            raise ValueError(
                f"MCP tool is not available for chat: {name}"
            )

        return await self._session.call_tool(name, arguments)

    async def create_approved_ticket(
        self,
        authorization: ApprovedTicketAuthorization,
    ) -> CallToolResult:
        if not isinstance(
            authorization,
            ApprovedTicketAuthorization,
        ):
            raise TypeError(
                "Ticket creation requires an approval authorization."
            )

        arguments = deepcopy(authorization.payload)
        arguments["idempotency_key"] = authorization.action_id

        return await self._session.call_tool(
            "create_ticket",
            arguments,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = Path(__file__).resolve().parent
    ticket_store = SQLiteTicketMetadataStore(configured_database_path())
    ticket_store.initialize()
    app.state.ticket_store = ticket_store
    approval_store = SQLiteApprovalStore(configured_database_path())
    approval_store.initialize()
    app.state.approval_store = approval_store
    memory_store = chat_memory.ChatMemoryStore(chat_memory.configured_memory_path())

    try:
        memory_store.initialize()
        app.state.memory_store = memory_store
    except chat_memory.ChatMemoryError:
        logger.exception("Chat memory is unavailable; continuing without it")
        app.state.memory_store = None

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "services.helpdesk_mcp"],
        env=dict(os.environ),
        cwd=root,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            available_tools = await session.list_tools()

            read_only_tools = tuple(
                tool
                for tool in available_tools.tools
                if tool.name in READ_ONLY_MCP_TOOL_NAMES
            )

            missing_tools = READ_ONLY_MCP_TOOL_NAMES - {
                tool.name for tool in read_only_tools
            }

            if missing_tools:
                raise RuntimeError(
                    "MCP server is missing required read-only tools: "
                    + ", ".join(sorted(missing_tools))
                )

            app.state.mcp = ReadOnlyMCPClient(
                session,
                read_only_tools,
            )

            yield


app = FastAPI(lifespan=lifespan)

# Development origins of the React demo only; no wildcard.
FRONTEND_ORIGINS = ["http://localhost:8443", "http://127.0.0.1:8443"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.state.pending_actions = {}
app.state.pending_actions_lock = Lock()
app.state.ticket_intakes = {}
app.state.last_cited_documents = {}


@dataclass
class PendingActionRecord:
    action: dict[str, Any]
    decision: bool | None = None
    write_status: str = "not_started"
    ticket_id: str | None = None
    local_recorded: bool = False


@dataclass
class TicketIntake:
    messages: list[str]
    requester_email: str | None = None


def _store_pending_action(
    session_id: str,
    action: dict[str, Any],
) -> None:
    action_id = action.get("action_id")

    if (
        not session_id.strip()
        or not isinstance(action_id, str)
        or not action_id.strip()
    ):
        raise ValueError(
            "A pending action requires a session ID and action ID."
        )

    key = (session_id, action_id)

    with app.state.pending_actions_lock:
        if key in app.state.pending_actions:
            raise ValueError(
                "This pending action already exists for the session."
            )

        record = PendingActionRecord(action=deepcopy(action))
        app.state.pending_actions[key] = record
        _persist_approval_state(app.state, session_id, record)


def _persist_approval_state(
    state: Any,
    session_id: str,
    record: PendingActionRecord,
) -> None:
    """Best-effort durable copy of approval state; never blocks the in-memory flow."""
    store = getattr(state, "approval_store", None)
    if store is None or record.action.get("test_only"):
        return
    try:
        store.save(
            session_id,
            record.action,
            decision=record.decision,
            write_status=record.write_status,
            ticket_id=record.ticket_id,
            local_recorded=record.local_recorded,
        )
    except ApprovalStoreError:
        logger.exception("Could not persist approval state.")


def _restore_approval_state(
    state: Any,
    session_id: str,
    action_id: str,
) -> PendingActionRecord | None:
    """Rebuild a record after a restart; only the owning session's hash can match."""
    store = getattr(state, "approval_store", None)
    if store is None:
        return None
    try:
        saved = store.load(session_id, action_id)
    except ApprovalStoreError:
        logger.exception("Could not read approval state.")
        return None
    if saved is None:
        return None
    return PendingActionRecord(
        action=saved["action"],
        decision=saved["decision"],
        write_status=saved["write_status"],
        ticket_id=saved["ticket_id"],
        local_recorded=saved["local_recorded"],
    )


class User(BaseModel):
    email: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    user: User | None = None


class ApproveRequest(BaseModel):
    session_id: str
    action_id: str
    approved: bool


@app.get("/health")
def health():
    return {"status": "ok"}


# --- Read-only ticket intelligence (disabled unless a token is configured) ---
# These routes only analyse existing tickets with deterministic local code. They make no model,
# embedding, MCP or network calls, never write, and are never invoked from /api/chat.
INTELLIGENCE_TOKEN_ENV = "HELPDESK_INTELLIGENCE_TOKEN"
DEFAULT_TICKET_DB_PATH = (
    Path(__file__).resolve().parent / "services" / "data" / "helpdesk_tickets.sqlite3"
)


def require_intelligence_access(request: Request) -> None:
    expected = (os.getenv(INTELLIGENCE_TOKEN_ENV) or "").strip()

    if not expected:
        raise HTTPException(status_code=404, detail="Not Found")

    scheme, _, supplied = (request.headers.get("authorization") or "").partition(" ")
    supplied = supplied.strip()
    authorized = scheme.lower() == "bearer" and hmac.compare_digest(
        supplied.encode("utf-8"), expected.encode("utf-8")
    )

    if not authorized:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )


class IntelligenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DuplicateCheckRequest(IntelligenceModel):
    subject: str = Field(min_length=1, max_length=300)
    description: str = Field(default="", max_length=4000)
    category: str = Field(min_length=1, max_length=64)
    requester_email: str | None = Field(default=None, max_length=254)
    created_at: str | None = Field(default=None, max_length=40)
    similarity_threshold: float = Field(
        default=ticket_intelligence.DEFAULT_SIMILARITY_THRESHOLD, ge=0.0, le=1.0
    )
    window_days: int | None = Field(
        default=ticket_intelligence.DEFAULT_WINDOW_DAYS, ge=1, le=3650
    )
    max_candidates: int = Field(default=ticket_intelligence.DEFAULT_MAX_CANDIDATES, ge=1, le=20)


class TrendRequest(IntelligenceModel):
    period: str = Field(default="month", pattern="^(week|month)$")
    group_by: str = Field(
        default="category", pattern="^(category|department|status|priority|error_code)$"
    )
    start: str | None = Field(default=None, max_length=40)
    end: str | None = Field(default=None, max_length=40)
    min_sample: int = Field(default=ticket_intelligence.DEFAULT_MIN_TREND_SAMPLE, ge=1, le=10000)


class KnowledgeProposalRequest(IntelligenceModel):
    min_tickets: int = Field(default=ticket_intelligence.DEFAULT_MIN_PROPOSAL_TICKETS, ge=1, le=100)
    max_proposals: int = Field(default=50, ge=1, le=200)


def _intelligence_ticket_db_path() -> Path:
    configured = (os.getenv("HELPDESK_TICKET_DB_PATH") or "").strip()
    return Path(configured) if configured else DEFAULT_TICKET_DB_PATH


def _load_intelligence_dataset() -> dict[str, Any]:
    try:
        return ticket_intelligence.load_ticket_dataset(db_path=_intelligence_ticket_db_path())
    except ticket_intelligence.IntelligenceDataError:
        logging.getLogger(__name__).error("Ticket intelligence data could not be read.")
        raise HTTPException(
            status_code=503, detail="Ticket data is currently unavailable."
        ) from None


def _intelligence_source(dataset: dict[str, Any]) -> dict[str, Any]:
    return {"ticket_counts": dataset["sources"], "read_only": True}


@app.post("/api/intelligence/duplicates", dependencies=[Depends(require_intelligence_access)])
def intelligence_duplicates(body: DuplicateCheckRequest):
    dataset = _load_intelligence_dataset()
    target = {
        "subject": body.subject,
        "description": body.description,
        "category": body.category,
        "requester_email": body.requester_email or "",
        "created_at": body.created_at,
    }
    result = ticket_intelligence.find_duplicate_candidates(
        target,
        dataset["tickets"],
        similarity_threshold=body.similarity_threshold,
        window_days=body.window_days,
        max_candidates=body.max_candidates,
    )
    # Candidates already carry only ticket IDs, status, reasons and match evidence.
    return {**result, "source": _intelligence_source(dataset)}


@app.post("/api/intelligence/trends", dependencies=[Depends(require_intelligence_access)])
def intelligence_trends(body: TrendRequest):
    dataset = _load_intelligence_dataset()

    for bound in (body.start, body.end):
        if bound is not None and ticket_intelligence._parse_timestamp(bound) is None:
            raise HTTPException(status_code=422, detail="start and end must be ISO 8601 timestamps.")

    result = ticket_intelligence.summarize_trends(
        dataset["tickets"],
        period=body.period,
        group_by=body.group_by,
        start=body.start,
        end=body.end,
        min_sample=body.min_sample,
        source="historical archive + ticket database",
    )
    return {**result, "source": _intelligence_source(dataset)}


@app.post(
    "/api/intelligence/knowledge-proposals",
    dependencies=[Depends(require_intelligence_access)],
)
def intelligence_knowledge_proposals(body: KnowledgeProposalRequest):
    dataset = _load_intelligence_dataset()

    try:
        catalog = ticket_intelligence.KnowledgeCatalog.load()
    except ticket_intelligence.IntelligenceDataError:
        logging.getLogger(__name__).error("Knowledge Base manifest could not be read.")
        raise HTTPException(
            status_code=503, detail="Knowledge Base metadata is currently unavailable."
        ) from None

    result = ticket_intelligence.propose_knowledge_updates(
        dataset["tickets"], catalog, min_tickets=body.min_tickets
    )

    proposals = []
    for proposal in result["proposals"][: body.max_proposals]:
        # Resolution text and per-ticket detail are withheld; source ticket IDs are kept for review.
        evidence = [
            {
                "ticket_id": item["ticket_id"],
                "status": item["status"],
                "resolved_at": item["resolved_at"],
                "kb_refs": item["kb_refs"],
            }
            for item in proposal["evidence"]
        ]
        proposals.append(
            {
                **{key: value for key, value in proposal.items() if key != "evidence"},
                "evidence": evidence,
                "resolution_text_included": False,
                "verified": False,
                "publishable": False,
                "requires_human_review": True,
            }
        )

    return {
        **{key: value for key, value in result.items() if key != "proposals"},
        "proposals": proposals,
        "proposals_total": len(result["proposals"]),
        "proposals_returned": len(proposals),
        "source": _intelligence_source(dataset),
    }


def _tool_result_data(
    tool_result: CallToolResult,
) -> tuple[dict[str, Any] | None, str | None]:
    if getattr(tool_result, "isError", False):
        return None, "The read-only helpdesk tool reported an error."

    data = getattr(tool_result, "structuredContent", None)

    if not isinstance(data, dict):
        content = getattr(tool_result, "content", None)
        if not isinstance(content, list):
            content = []

        text_result = next(
            (
                item.text
                for item in content
                if getattr(item, "type", None) == "text"
                and isinstance(getattr(item, "text", None), str)
            ),
            None,
        )

        try:
            data = (
                json.loads(text_result)
                if text_result is not None
                else None
            )
        except json.JSONDecodeError:
            data = None

    if not isinstance(data, dict):
        return None, "The read-only helpdesk tool returned invalid data."

    if data.get("ok") is not True:
        return None, "The read-only helpdesk tool could not complete the request."

    return data, None


NAKHLA_EMAIL_DOMAIN = "nakhla.example"

# Practical address syntax (local part, single @, dotted domain). It checks
# format only; it says nothing about who owns the address.
EMAIL_PATTERN = (
    r"[a-z0-9._%+-]+@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}"
)
_EMAIL_FORMAT = re.compile(rf"{EMAIL_PATTERN}", re.IGNORECASE)


def _is_valid_email_format(email: Any) -> bool:
    return (
        isinstance(email, str)
        and len(email) <= 254
        and ".." not in email
        and _EMAIL_FORMAT.fullmatch(email.strip()) is not None
    )


def _is_nakhla_email(email: str) -> bool:
    return email.strip().lower().endswith(f"@{NAKHLA_EMAIL_DOMAIN}")


def _email_from_message(message: str) -> str | None:
    match = re.search(
        rf"(?<![\w.%+-]){EMAIL_PATTERN}(?![\w-])",
        message,
        re.IGNORECASE,
    )
    return match.group(0).lower() if match else None


def _claimed_department_in_message(message: str) -> str | None:
    """Extract an explicitly self-claimed department without treating it as verified."""
    departments = TICKET_SCHEMA.get("$defs", {}).get("department", {}).get(
        "enum", []
    )

    for department in sorted(departments, key=len, reverse=True):
        aliases = [str(department)]
        if str(department).casefold() == "contractors":
            aliases.append("contractor")

        department_pattern = "|".join(
            re.escape(alias) for alias in aliases
        )
        claim = re.search(
            rf"\b(?:i\s+am|i'm|i\s+work\s+(?:in|for)|"
            rf"my\s+department\s+is)\s+"
            rf"(?:(?:a|an)\s+)?(?:(?:in|from)\s+)?(?:the\s+)?"
            rf"({department_pattern})\b",
            _unquoted_user_text(message),
            re.IGNORECASE,
        )

        if claim:
            return str(department)

    return None


def _requester_identity_conflicts(
    intake: TicketIntake,
    request: ChatRequest,
    employee: dict[str, Any],
) -> bool:
    """Reject explicit identity claims that conflict with directory lookup."""
    verified_email = str(employee.get("email", "")).strip().lower()
    claimed_emails: set[str] = set()

    if request.user and request.user.email.strip():
        claimed_emails.add(request.user.email.strip().lower())

    for message in intake.messages:
        clean = _unquoted_user_text(message)
        claimed_emails.update(
            match.group(0).lower()
            for match in _EMAIL_FORMAT.finditer(clean)
            if _is_nakhla_email(match.group(0))
        )

    if any(email != verified_email for email in claimed_emails):
        return True

    departments = TICKET_SCHEMA.get("$defs", {}).get("department", {}).get(
        "enum", []
    )
    department_pattern = "|".join(
        re.escape(value)
        for value in sorted(departments, key=len, reverse=True)
    )
    department_claims = re.findall(
        rf"\b(?:from|in|my department is|i work (?:in|for))\s+"
        rf"(?:the\s+)?({department_pattern})(?:\s+department)?\b",
        _unquoted_user_text(_ticket_report(intake.messages)),
        re.IGNORECASE,
    )
    verified_department = str(employee.get("department", "")).strip()

    claimed_department = _claimed_department_in_message(
        _ticket_report(intake.messages)
    )
    if (
        claimed_department
        and claimed_department.casefold() != verified_department.casefold()
    ):
        return True

    if any(
        claimed.casefold() != verified_department.casefold()
        for claimed in department_claims
    ):
        return True

    name_pattern = re.compile(
        r"\b(?:my name is|i am|i'm)\s+"
        r"([A-Z][A-Za-z'’-]*(?:\s+[A-Z][A-Za-z'’-]*){0,2})"
        r"(?=\s+(?:from|in|at|and|but|because|with|cannot|can't|"
        r"have|need|work|unable|having)\b|[,.;!?]|$)"
    )
    claimed_names = [
        " ".join(match.group(1).casefold().split())
        for match in name_pattern.finditer(
            _unquoted_user_text(_ticket_report(intake.messages))
        )
    ]
    directory_name = str(
        employee.get("name")
        or employee.get("full_name")
        or " ".join(
            str(employee.get(field, "")).strip()
            for field in ("first_name", "last_name")
            if employee.get(field)
        )
    )
    directory_name_parts = re.findall(r"[a-z]+", directory_name.casefold())

    if claimed_names:
        if not directory_name_parts:
            return True

        for claimed_name in claimed_names:
            claimed_parts = re.findall(r"[a-z]+", claimed_name)
            if claimed_parts and claimed_parts != directory_name_parts[:len(claimed_parts)]:
                return True

    return False


def _ticket_report(messages: list[str]) -> str:
    return "\n".join(message.strip() for message in messages if message.strip())


def _unquoted_user_text(message: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " ", message)
    text = re.sub(r"(?m)^\s*>.*$", " ", text)

    def preserve_error_code(match: re.Match[str]) -> str:
        quoted_value = next(
            (
                group
                for group in match.groups()
                if group is not None
            ),
            match.group(0)[1:-1],
        ).strip()

        if re.fullmatch(
            r"[A-Z]{2,5}-\d{3}",
            quoted_value,
            re.IGNORECASE,
        ):
            return quoted_value.upper()

        return " "

    return re.sub(
        r""" "(?:\\.|[^"\\])*" | “([^”]*)” | ‘([^’]*)’ | """
        r"""(?<!\w)'([^'\n]*(?:'[a-z][^'\n]*)*)'(?!\w) """,
        preserve_error_code,
        text,
        flags=re.VERBOSE | re.IGNORECASE,
    )


def _attempted_steps(messages: list[str]) -> list[str]:
    steps: list[str] = []
    seen: set[str] = set()

    def add_step(value: str) -> None:
        # Impact statements ("only me, I'm blocked") are not troubleshooting steps.
        value = re.split(
            r"[,;]?\s+(?:and\s+)?(?:only\s+(?:i|me)\b|i['’]?m\s+(?:blocked|affected)|"
            r"i\s+am\s+(?:blocked|affected)|(?:there\s+is\s+)?no\s+workaround|"
            r"(?:\d+|several|many|all)\s+(?:users|people|colleagues)\b)",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        step = " ".join(value.strip(" \t,;:-").split())

        if (
            len(step) >= 3
            and "ticket" not in step.lower()
            and step.casefold() not in seen
        ):
            steps.append(step)
            seen.add(step.casefold())

    action_verbs = (
        "restarted|rebooted|checked|updated|reinstalled|installed|retried|"
        "reset|switched|reconnected|cleared|tested|ran|followed|"
        "synced|contacted|toggled|changed|opened|connected|"
        "disconnected|removed|repaired|replaced|uninstalled"
    )

    for raw_message in messages:
        message = _unquoted_user_text(raw_message)
        for line in message.splitlines():
            labeled = re.search(
                r"\b(?:steps?\s+(?:already\s+)?tried|attempted\s+steps|tried)"
                r"\s*:\s*(.+)",
                line,
                re.IGNORECASE,
            )

            did_all = re.search(
                r"\b(?:i|we)\s+(?:did|followed|completed)\s+"
                r"(?:all\s+of\s+(?:that|these|them)|everything|"
                r"all\s+(?:the\s+)?steps)\s*:\s*([^.!?\n]+)",
                line,
                re.IGNORECASE,
            )

            if did_all:
                for item in re.split(
                    r"\s*,\s*(?:and\s+)?|\s+and\s+", did_all.group(1)
                ):
                    add_step(item)
                continue

            if labeled:
                for item in re.split(r"\s*;\s*", labeled.group(1)):
                    add_step(item)
                continue

            tried = re.search(
                r"\b(?:i|we)(?:'ve|’ve| have| had)?\s+"
                r"(?:(?:already|also)\s+)*(?:tried|attempted)\s+"
                r"(?:to\s+)?([^.!?\n]+)",
                line,
                re.IGNORECASE,
            )

            if tried:
                add_step(tried.group(1))

            performed = re.search(
                rf"\b(?:i|we)(?:'ve|’ve| have| had)?\s+"
                rf"(?:(?:already|also)\s+)*({action_verbs})\s+"
                r"([^.!?\n]+)",
                line,
                re.IGNORECASE,
            )

            if performed:
                first_verb = performed.group(1)
                tail = performed.group(2)
                action_pattern = re.compile(
                    rf"\b({action_verbs})\s+(.+?)"
                    rf"(?=\s+and\s+(?:{action_verbs})\b|[,!?\.]|$)",
                    re.IGNORECASE,
                )
                later_actions = list(action_pattern.finditer(tail))

                if later_actions:
                    first_action_start = later_actions[0].start()
                    first_object = re.split(
                        r"\s+and\s+$", tail[:first_action_start], maxsplit=1,
                        flags=re.IGNORECASE,
                    )[0].strip(" ,;:-")
                    if first_object:
                        add_step(f"{first_verb} {first_object}")
                    for action in later_actions:
                        add_step(f"{action.group(1)} {action.group(2).strip()}")
                else:
                    add_step(f"{first_verb} {tail}")

            standalone = re.match(
                rf"\s*({action_verbs})\s+([^.!?\n]+)",
                line,
                re.IGNORECASE,
            )

            if standalone:
                add_step(
                    f"{standalone.group(1)} {standalone.group(2)}"
                )

    return steps


def _extract_affected_users(report: str) -> int | None:
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    match = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s+(?:users|people|employees|colleagues)\b",
        report,
        re.IGNORECASE,
    )

    if match:
        value = match.group(1).lower()
        return int(value) if value.isdigit() else number_words[value]

    if re.search(
        r"\b(?:only\s+me|just\s+me|only\s+myself|just\s+myself)\b|"
        r"\b(?:only|just)\s+(?:i|i\s+am|i'm|im)\s+(?:am\s+)?affected\b|"
        r"\b(?:i am|i'm|we are)\s+the\s+only\s+"
        r"(?:affected\s+)?(?:user|person|employee|colleague)\b|"
        r"\bthe\s+only\s+affected\s+(?:user|person|employee|colleague)\b",
        report,
        re.IGNORECASE,
    ):
        return 1

    return None


def _ticket_category(report: str, department: str) -> str:
    lowered = report.lower()

    if re.search(
        r"\bphish(?:ing)?\b|"
        r"\b(?:lost|stolen)\s+(?:company\s+)?"
        r"(?:phone|laptop|device)\b|"
        r"\b(?:suspicious sign-?in|security incident)\b",
        lowered,
    ):
        return "security"

    if (
        department == "Finance"
        and re.search(r"\b(?:vpn|nakhlaconnect|remote access)\b", lowered)
    ):
        return "security"

    if re.search(
        r"\b(?:vpn|nakhlaconnect|wi-?fi|network share|vdi|"
        r"network access|company network|corporate network)\b",
        lowered,
    ):
        return "network_access"

    if re.search(
        r"\b(?:password|mfa|nakhlakey|nakhlaid|sign[ -]?in|"
        r"sso|account lock|permission|access request)\b",
        lowered,
    ):
        return "identity_access"

    if re.search(
        r"\b(?:laptop|computer|dock|printer|phone|device|"
        r"nakhlamanage)\b",
        lowered,
    ):
        # "updated the firmware from Software Center" is a step, not the topic.
        topic = re.sub(r"\bfrom\s+(?:the\s+)?software center\b", " ", lowered)

        if not re.search(
            r"\b(?:software center|software install|installation|"
            r"license|licence|application install)\b",
            topic,
        ):
            return "hardware"

    if re.search(
        r"\b(?:software center|software install|installation|"
        r"license|licence|application install)\b",
        lowered,
    ):
        return "software"

    if re.search(
        r"\b(?:email|e-mail|mailbox|calendar|nakhla mail)\b",
        lowered,
    ):
        return "email_collab"

    if re.search(
        r"\b(?:onboarding|new starter|offboarding|leaver)\b",
        lowered,
    ):
        return "onboarding"

    return "other"


def _ticket_priority(
    report: str,
    department: str,
    affected_users: int | None,
) -> str | None:
    lowered = report.lower()

    if department == "Executive Office":
        return "P1"

    if affected_users is not None and affected_users > 3:
        return "P1"

    if re.search(
        r"\b(?:lost|stolen)\s+(?:company\s+)?"
        r"(?:phone|laptop|device)\b|"
        r"\b(?:entered|typed)\s+(?:my\s+)?password\s+"
        r"(?:into|on)\s+(?:a\s+)?phish",
        lowered,
    ):
        return "P1"

    is_request = bool(
        re.search(
            r"\b(?:access request|request access|how do i|how to|"
            r"please install|need an? (?:install|access|permission)|"
            r"software request|new starter|onboarding request)\b",
            lowered,
        )
    )

    if is_request:
        return "P4"

    no_workaround = bool(
        re.search(
            r"\b(?:no|without|not have|don't have|do not have)\s+"
            r"(?:an?\s+)?(?:available\s+)?(?:workaround|alternative)\b|"
            r"\bthere\s+(?:is|are)\s+no\s+"
            r"(?:available\s+)?(?:workaround|alternative)\b|"
            r"\bthere's\s+no\s+(?:available\s+)?"
            r"(?:workaround|alternative)\b",
            lowered,
        )
    )
    has_workaround = bool(
        re.search(
            r"\b(?:workaround|alternative|can still work|"
            r"still able to work)\b",
            lowered,
        )
    ) and not no_workaround

    # KB-016: Finance VPN problems are P2 (P1 above 3 users, handled above).
    if department == "Finance" and re.search(
        r"\b(?:vpn|nakhlaconnect|remote access)\b", lowered
    ):
        return "P2"

    # KB-002 explicitly assigns P2 when a requester cannot connect with
    # NX-403. The Executive Office and wider-impact P1 rules run above.
    if re.search(r"\bnx-?403\b", lowered) and _ticket_service(lowered) == "vpn":
        return "P2" if affected_users is not None else None

    # The requester is themselves blocked; a larger count is only assumed
    # when the user states one (more than 3 is P1 above).
    completely_blocked = bool(
        re.search(
            r"\bcompletely blocked(?:\s+from working)?\b|"
            r"\b(?:completely|fully)\s+blocks?\s+(?:me|my work)\b|"
            r"\bmy work is completely blocked\b|"
            r"\b(?:my|our|the)\s+work is blocked\b|"
            r"\b(?:i am|i'm|im)\s+(?:completely\s+)?blocked\b|"
            r"\b(?:cannot|can't|unable to)\s+work\b",
            lowered,
        )
    )

    if affected_users is None:
        return None

    if affected_users <= 3 and completely_blocked and no_workaround:
        return "P2"

    if has_workaround or re.search(
        r"\bdegraded but usable\b",
        lowered,
    ):
        return "P3"

    return None


def _ticket_service(report: str) -> str | None:
    service_terms = (
        (r"\b(?:vpn|nakhlaconnect)\b", "vpn"),
        (r"\bwi-?fi\b", "wifi"),
        (r"\b(?:nakhlamail|nakhla mail|mailbox|email)\b", "mail"),
        (r"\b(?:nakhlaid|sso|sign[ -]?in)\b", "nakhlaid"),
        (r"\b(?:nakhlaprint|printing|printer)\b", "print"),
        (r"\b(?:software center|software catalogue|software catalog)\b", "software"),
        (r"\b(?:vdi|nakhlaworkspace)\b", "vdi"),
        (r"\b(?:nakhlamanage|device enrollment|device compliance)\b", "mdm"),
    )

    for pattern, service in service_terms:
        if re.search(pattern, report, re.IGNORECASE):
            return service

    return None


def _active_matching_incident(
    status_data: dict[str, Any],
    report: str,
    service: str | None,
) -> dict[str, Any] | None:
    status = status_data.get("status")

    if not isinstance(status, dict):
        return None

    incidents = status.get("incidents")

    if not isinstance(incidents, list):
        return None

    issue_codes = set(kb_precedence.normalize_error_codes(report))

    for incident in incidents:
        if (
            not isinstance(incident, dict)
            or str(incident.get("status", "")).lower() != "active"
            or incident.get("service") != service
            or service is None
        ):
            continue

        incident_codes = incident.get("error_codes")

        if (
            isinstance(incident_codes, list)
            and issue_codes.intersection(
                str(code).upper() for code in incident_codes
            )
        ):
            return incident

    return None


_SERVICE_STATUS_QUESTION = re.compile(
        r"\b(?:status|active incident|outage|operational|currently down|"
        r"currently working|working right now|available right now)\b",
        re.IGNORECASE,
)


def _is_service_status_question(message: str) -> bool:
        if re.search(r"\bticket\b.{0,30}\bstatus\b|\bstatus\b.{0,30}\bticket\b", message, re.IGNORECASE):
            return False
        if re.search(r"\bIT-\d{3,6}\b", message, re.IGNORECASE):
            return False
        return bool(_SERVICE_STATUS_QUESTION.search(message))


def _validated_status_page(
        tool_result: CallToolResult,
) -> tuple[dict[str, Any] | None, str | None]:
        data, error_message = _tool_result_data(tool_result)
        if error_message or data is None:
            return None, error_message or "The live status tool returned no data."

        status = data.get("status")
        if (
            not isinstance(status, dict)
            or not isinstance(status.get("services"), list)
            or not isinstance(status.get("incidents"), list)
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("id"), str)
                or not isinstance(item.get("status"), str)
                for item in status["services"]
            )
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("id"), str)
                or not isinstance(item.get("service"), str)
                or not isinstance(item.get("status"), str)
                or (
                    "error_codes" in item
                    and (
                        not isinstance(item["error_codes"], list)
                        or not all(isinstance(code, str) for code in item["error_codes"])
                    )
                )
                for item in status["incidents"]
            )
        ):
            return None, "The live status tool returned malformed data."

        return data, None


def _status_value(value: Any) -> str:
        return " ".join(str(value).split())[:300] if isinstance(value, str) else ""


async def _live_service_status_response(message: str) -> dict[str, Any]:
        service = _ticket_service(message)
        label = {
            "vpn": "VPN",
            "wifi": "Wi-Fi",
            "nakhlaid": "NakhlaID",
            "mail": "Nakhla Mail",
            "print": "NakhlaPrint",
            "software": "Software Center",
            "vdi": "VDI",
            "mdm": "NakhlaManage",
        }.get(service, "IT services")

        try:
            tool_result = await app.state.mcp.call_tool("get_service_status", {})
        except Exception:
            logger.exception("Live service status lookup failed")
            status_data, status_error = None, "the live status tool could not be reached"
        else:
            status_data, status_error = _validated_status_page(tool_result)

        tool_calls = [{"tool": "get_service_status", "ok": status_error is None}]
        if status_error or status_data is None:
            reply = (
                f"I could not check the live {label} status or verify whether "
                "an incident is active."
            )
            return {
                "reply": reply,
                "pending_action": None,
                "citations": [],
                "tool_calls": tool_calls,
                "result": {},
                "state": "service_status_unavailable",
            }

        status = status_data["status"]
        services = status["services"]
        matching_services = [
            item
            for item in services
            if service is None
            or str(item.get("id", "")).casefold() == service.casefold()
            or service.casefold() in str(item.get("name", "")).casefold()
        ]

        if service is not None and not matching_services:
            return {
                "reply": (
                    f"The live status page returned no entry for {label}; "
                    "I cannot confirm its current status."
                ),
                "pending_action": None,
                "citations": [],
                "tool_calls": tool_calls,
                "result": {},
                "state": "service_status_unavailable",
            }

        lines = ["Live service status:"]
        for item in matching_services:
            name = _status_value(item.get("name")) or _status_value(item.get("id"))
            current = _status_value(item.get("status"))
            lines.append(f"- {name}: {current}.")

        service_ids = {
            str(item.get("id", "")).casefold()
            for item in matching_services
        }
        active_incidents = [
            incident
            for incident in status["incidents"]
            if str(incident.get("status", "")).casefold() == "active"
            and (service is None or str(incident.get("service", "")).casefold() in service_ids)
        ]

        if active_incidents:
            lines.append("Active incidents:")
            for incident in active_incidents:
                title = _status_value(incident.get("title"))
                detail = f"{incident['id']}"
                if title:
                    detail += f" — {title}"
                codes = incident.get("error_codes")
                if isinstance(codes, list) and codes:
                    detail += f" (error codes: {', '.join(codes)})"
                lines.append(f"- {detail}.")
                workaround = _status_value(incident.get("workaround"))
                instructions = _status_value(incident.get("instructions"))
                if workaround:
                    lines.append(f"  Workaround: {workaround}")
                if instructions:
                    lines.append(f"  Status-page instructions: {instructions}")
        else:
            lines.append(f"No active {label} incident was listed by the status tool.")

        return {
            "reply": "\n".join(lines),
            "pending_action": None,
            "citations": [],
            "tool_calls": tool_calls,
            "result": {},
            "state": "service_status",
        }

def _remember_cited_documents(session_id: str, citations: Any) -> None:
    """Keep the verified citations of this session's last answer (bounded)."""
    cited = app.state.last_cited_documents

    ids = [
        item["id"]
        for item in citations or []
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]

    if ids:
        cited[session_id] = ids
    else:
        cited.pop(session_id, None)

    while len(cited) > 500:
        cited.pop(next(iter(cited)))


_FOLLOWED_GUIDANCE = re.compile(
    r"\bI\s+(?:did|followed|completed|tried)\s+(?:all\s+of\s+)?"
    r"(?:that|those|them|these|everything|the\s+steps)\b",
    re.IGNORECASE,
)


def _followed_guidance_refs(session_id: str, messages: list[str]) -> list[str]:
    """KB ids the assistant verifiably cited, when the user says they followed it."""
    if not any(
        _FOLLOWED_GUIDANCE.search(_unquoted_user_text(m)) for m in messages
    ):
        return []

    return list(app.state.last_cited_documents.get(session_id, []))


def _extract_kb_refs(messages: list[str]) -> list[str]:
    known_ids = {
        str(document.get("id"))
        for document in json.loads(
            (Path(__file__).resolve().parent / "dataset" / "manifest.json")
            .read_text(encoding="utf-8")
        ).get("documents", [])
        if isinstance(document, dict) and document.get("id")
    }
    refs: list[str] = []

    for message in messages:
        message = _unquoted_user_text(message)
        for match in re.finditer(
            r"\b(?:KB|SP)-\d{3}\b",
            message,
            re.IGNORECASE,
        ):
            context = message[max(0, match.start() - 70):match.start()]
            reference = match.group(0).upper()

            if (
                reference in known_ids
                and re.search(
                    r"\b(?:followed|used|tried|read)\b",
                    context,
                    re.IGNORECASE,
                )
                and reference not in refs
            ):
                refs.append(reference)

    return refs


def _classification_kb_refs(
    category: str | None,
    priority: str | None,
    department: str,
    report: str,
) -> list[str]:
    """Return eligible policy sources that support the proposed classification."""
    candidate_ids: list[str] = []
    if category or priority:
        candidate_ids.append("KB-015")

    department_text = report.lower()
    if department == "Executive Office" or (
        department == "Finance"
        and re.search(
            r"\b(?:vpn|nakhlaconnect|remote access|connect from home)\b",
            department_text,
        )
    ) or (department == "Contractors" and _ticket_service(report) == "vdi"):
        candidate_ids.append("KB-016")

    policy = kb_precedence.load_policy()
    return [
        document_id
        for document_id in dict.fromkeys(candidate_ids)
        if kb_precedence.is_eligible(
            {
                "id": document_id,
                "status": policy.documents.get(document_id, {}).get("status"),
            },
            policy,
        )
    ]


def _ticket_subject(report: str) -> str:
    subject_text = re.sub(
        rf"(?<![\w.%+-]){EMAIL_PATTERN}(?![\w-])",
        "",
        report,
        flags=re.IGNORECASE,
    )

    for pattern in TICKET_CREATION_PATTERNS:
        subject_text = pattern.sub("", subject_text)

    subject_text = re.sub(
        r"^\s*(?:hi|hello|hey)\b[!,.:\s-]*",
        "",
        subject_text,
        flags=re.IGNORECASE,
    )
    department_pattern = "|".join(
        re.escape(value)
        for value in sorted(
            TICKET_SCHEMA.get("$defs", {}).get("department", {}).get("enum", []),
            key=len,
            reverse=True,
        )
    )
    subject_text = re.sub(
        rf"^\s*(?:i am|i'm|my name is)\s+[^,.;!?]{{1,80}}?\s+"
        rf"(?:from|in)\s+(?:the\s+)?(?:{department_pattern})"
        r"(?:\s+department)?\s*[,;:]\s*",
        "",
        subject_text,
        flags=re.IGNORECASE,
    )
    subject_text = re.sub(
        r"\b(?:please|for me|i would like to|i'd like to)\b",
        "",
        subject_text,
        flags=re.IGNORECASE,
    )
    fragments = re.split(r"[,;.!?\n]", subject_text)
    candidate = next(
        (
            " ".join(fragment.split()).strip(" :,-")
            for fragment in fragments
            if len(" ".join(fragment.split()).strip(" :,-")) >= 5
            and not re.match(
                r"^(?:i am|i'm)\s+in\s+(?:the\s+)?(?:finance|engineering|hr|sales|support|executive office|contractors)(?:\s+department)?$",
                fragment.strip(),
                re.IGNORECASE,
            )
            and not re.match(
                r"^(?:i am|i'm|my name is)\s+"
                r"[A-Z][A-Za-z'’-]*(?:\s+[A-Z][A-Za-z'’-]*){1,2}$",
                fragment.strip(),
            )
            and not re.match(
                r"^(?:i|we)\s+(?:have\s+)?(?:tried|attempted)\b",
                fragment.strip(),
                re.IGNORECASE,
            )
        ),
        "",
    )
    candidate = re.sub(
        r"^(?:i\s+)?(?:cannot|can't)\s+",
        "Unable to ",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(
        r"^(?:i\s+)?am\s+unable\s+to\s+",
        "Unable to ",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(
        r"^Unable to access (?:the\s+)?company network$",
        "Unable to access the company network",
        candidate,
        flags=re.IGNORECASE,
    )

    code_match = re.search(
        r"\b(?!KB-|SP-)([A-Z]{2,5})-?(\d{3,4})\b",
        report,
        re.IGNORECASE,
    )
    if code_match:
        normalized_code = f"{code_match.group(1).upper()}-{code_match.group(2)}"
        if normalized_code not in candidate.upper():
            suffix = f" ({normalized_code})"
            if len(candidate) + len(suffix) > 120:
                candidate = candidate[: 120 - len(suffix) - 3].rstrip(" .,:;-") + "..."
            candidate += suffix

    if len(candidate) > 120:
        shortened = candidate[:117].rstrip()
        if " " in shortened:
            shortened = shortened.rsplit(" ", 1)[0]
        candidate = shortened + "..."

    return candidate


def _kb_direct_escalation_step(
    report: str,
    department: str,
    service: str | None,
    *,
    status_checked: bool = False,
) -> tuple[str, str] | None:
    """A step the KB itself prescribes when the user has nothing to try.

    Returns (attempted step, KB id) only for documented cases: Finance VPN
    (KB-016) or an error code that a current article says has no user-side
    fix. Nothing is invented when the KB says otherwise.
    """
    if department == "Finance" and service == "vpn":
        if status_checked:
            step = (
                "Checked the service status; no matching incident was listed. "
                "No client troubleshooting was performed under KB-016."
            )
        else:
            step = (
                "No troubleshooting steps were taken; Finance VPN issues go "
                "straight to Security Operations under KB-016."
            )
        return (
            step,
            "KB-016",
        )

    codes = set(kb_precedence.normalize_error_codes(report))

    if not codes:
        return None

    kb_dir = Path(__file__).resolve().parent / "dataset" / "knowledge_base"

    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        doc_id = re.search(r"^id:\s*(\S+)\s*$", text, re.MULTILINE)

        if not doc_id or not re.search(
            r"^status:\s*current\s*$", text, re.MULTILINE
        ):
            continue

        for line in text.splitlines():
            if "nothing to fix on your side" not in line.lower():
                continue

            for code in sorted(codes):
                if code in line.upper():
                    return (
                        f"No user-side fix exists for {code}; "
                        f"{doc_id.group(1)} says to open a ticket",
                        doc_id.group(1),
                    )

    return None


def _confirms_escalation(session_id: str, message: str) -> bool:
    """A bare "yes" counts only right after the assistant offered escalation."""
    if not isinstance(message, str) or not _AFFIRMATION.match(message):
        return False

    store = _memory_store()

    if store is None:
        return False

    try:
        history = store.history(session_id)
    except chat_memory.ChatMemoryError:
        return False

    return bool(
        history
        and history[-1]["role"] == "assistant"
        and _ESCALATION_QUESTION.search(history[-1]["content"])
    )


def _begin_or_continue_ticket_intake(
    request: ChatRequest,
) -> TicketIntake | None:
    with app.state.pending_actions_lock:
        current = app.state.ticket_intakes.get(request.session_id)

        if _requests_ticket_creation(request.message):
            # Only a message with no issue of its own borrows the previous one;
            # a new issue must not inherit an unrelated earlier topic.
            earlier = (
                _previous_user_turn(request.session_id)
                if _is_bare_ticket_request(request.message)
                else []
            )
            intake = TicketIntake(messages=[*earlier, request.message])
        elif current is None and _confirms_escalation(
            request.session_id, request.message
        ):
            intake = TicketIntake(
                messages=[
                    *_previous_user_turn(request.session_id),
                    request.message,
                ]
            )
        elif current is not None and _leaves_ticket_intake(request.message):
            app.state.ticket_intakes.pop(request.session_id, None)
            return None
        elif current is not None:
            intake = current
            intake.messages.append(request.message)
        else:
            return None

        supplied_email = (
            request.user.email.strip().lower()
            if request.user and request.user.email.strip()
            else _email_from_message(
                _unquoted_user_text(request.message)
            )
        )

        if supplied_email:
            intake.requester_email = supplied_email

        app.state.ticket_intakes[request.session_id] = intake
        return TicketIntake(
            messages=list(intake.messages),
            requester_email=intake.requester_email,
        )


def _clear_ticket_intake(session_id: str) -> None:
    with app.state.pending_actions_lock:
        app.state.ticket_intakes.pop(session_id, None)


MAX_CHAT_HISTORY_MESSAGES = chat_memory.MAX_HISTORY_MESSAGES


def _memory_store() -> chat_memory.ChatMemoryStore | None:
    return getattr(app.state, "memory_store", None)


def _previous_user_turn(session_id: str) -> list[str]:
    """Return recent issue context while skipping bare escalation commands."""

    store = _memory_store()

    if store is None:
        return []

    try:
        users = [
            item["content"]
            for item in store.history(session_id)
            if item["role"] == "user"
        ]
    except chat_memory.ChatMemoryError:
        return []

    recent = users[-8:]
    relevant = [
        item[:1000]
        for item in recent
        if not _is_bare_ticket_request(item)
    ]
    return relevant[-4:]


def _chat_history(
    session_id: str,
    message: str = "",
    owner: str | None = None,
    ticket_lines: list[str] | None = None,
) -> list[dict[str, str]]:
    """Recent turns plus relevant long-term memory.

    History is per session. Facts, notes and earlier conversations follow the
    directory-verified employee (`owner`) across sessions; without one they
    fall back to the session.
    """

    store = _memory_store()

    if store is None:
        return []

    try:
        store.save_facts(session_id, chat_memory.extract_facts(message), owner)
        history = store.history(session_id)
        facts = chat_memory.relevant_facts(store.facts(session_id, owner), message)
        notes = store.relevant_memories(
            session_id,
            message,
            exclude_issues={
                item["content"] for item in history if item["role"] == "user"
            },
            owner=owner,
        )
        episodes = store.past_episodes(
            session_id, owner, message, new_conversation=not history
        )
    except chat_memory.ChatMemoryError:
        logger.exception("Chat memory could not be read")
        return []

    context = chat_memory.memory_context_message(
        notes, facts, episodes, ticket_lines
    )

    return ([context] if context else []) + history


def _remember_intake_turn(session_id: str, message: str, reply: Any) -> None:
    """Record a ticket-intake exchange so a later "yes" is judged against it.

    Unlike _remember_chat_turn this never confirms or stages a resolution.
    """

    store = _memory_store()

    if store is None or not isinstance(reply, str) or not reply.strip():
        return

    try:
        store.add_turn(session_id, message, reply)
    except chat_memory.ChatMemoryError:
        logger.exception("Chat memory could not be stored")


def _remember_chat_turn(
    session_id: str,
    message: str,
    reply: str,
    citations: list[Any] | None = None,
    owner: str | None = None,
) -> None:
    """Persist the user message and final verified reply; never fail the chat.

    A resolution becomes a long-term note only when the USER confirms that the
    previous verified, cited answer worked.
    """

    store = _memory_store()

    if store is None:
        return

    try:
        if chat_memory.is_user_confirmation(message):
            store.confirm_resolution(session_id, owner)

        store.add_turn(session_id, message, reply)
        store.stage_resolution(session_id, message, reply, citations)
        if not is_memory_chatter(message) and not _TICKET_ID.search(message):
            store.add_episode(session_id, owner, message, reply, citations)
    except chat_memory.ChatMemoryError:
        logger.exception("Chat memory could not be stored")


class ForgetRequest(BaseModel):
    session_id: str
    email: str | None = None


class IdentifyRequest(BaseModel):
    email: str


TICKET_LIST_FIELDS = (
    "ticket_id",
    "subject",
    "category",
    "priority",
    "status",
    "department",
    "created_at",
)


_TICKET_ID = re.compile(r"\bIT-\d{3,6}\b", re.IGNORECASE)
_WHO_AM_I = re.compile(
    r"^\W*(?:what(?:'s|\s+is)\s+my\s+name|who\s+am\s+i|what\s+am\s+i\s+called)\b",
    re.IGNORECASE,
)
_RECURRENCE_HINT = re.compile(
    r"\b(?:again|still|before|previous(?:ly)?|last time|same (?:issue|problem)|"
    r"earlier|recurring|keeps?|ticket|remember)\b",
    re.IGNORECASE,
)
_TICKET_STATUS_INTENT = re.compile(
    r"\b(?:status|update|progress|resolved|resolution|closed|open|happened|"
    r"what(?:'s| is) going on|any news|check|look up|details?)\b",
    re.IGNORECASE,
)


def _single_line(value: Any, limit: int = 90) -> str:
    return " ".join(str(value).split())[:limit] if isinstance(value, str) else ""


async def _recent_ticket_lines(email: str, limit: int = 3) -> list[str]:
    """Short facts about the employee's latest tickets from the system of record.

    Read-only (search_tickets). Only fixed fields are used and the subject is
    shortened and secret-checked; ticket text is evidence, never instructions.
    """
    try:
        result = await app.state.mcp.call_tool(
            "search_tickets", {"requester_email": email, "limit": limit}
        )
    except Exception:
        logger.exception("Ticket history lookup failed")
        return []

    data, error_message = _tool_result_data(result)
    rows = data.get("tickets") if data and not error_message else None

    if not isinstance(rows, list):
        return []

    lines = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or str(row.get("requester_email", "")).strip().lower() != email
            or not isinstance(row.get("ticket_id"), str)
        ):
            continue
        subject = _single_line(row.get("subject"))
        if chat_memory.contains_secret(subject):
            subject = "(hidden)"
        lines.append(
            f"{row['ticket_id']} [{_single_line(row.get('status'), 20)}, "
            f"{_single_line(row.get('category'), 30)}] {subject}"
        )

    return lines[:limit]


async def _ticket_lookup_response(
    message: str, signed_in_email: str | None
) -> dict[str, Any] | None:
    """Answer 'what is the status of IT-2016?' through the read-only get_ticket tool.

    Returns None when the message is not such a question. Only the signed-in,
    directory-verified requester may see a ticket; other tickets are never shown.
    """
    ticket_ids = list(dict.fromkeys(
        match.upper() for match in _TICKET_ID.findall(_unquoted_user_text(message))
    ))

    if not ticket_ids or not _TICKET_STATUS_INTENT.search(message):
        return None

    def reply(text: str, state: str, calls: list[dict[str, Any]] | None = None):
        return {
            "reply": text,
            "pending_action": None,
            "citations": [],
            "tool_calls": calls or [],
            "result": {},
            "state": state,
        }

    employee = None
    if signed_in_email:
        employee, _ = await _lookup_employee(signed_in_email)

    if employee is None:
        return reply(
            "Please sign in with your work email first, so I can confirm the "
            "ticket is yours before showing it.",
            "ticket_lookup_requires_identity",
        )

    ticket_id = ticket_ids[0]

    try:
        tool_result = await app.state.mcp.call_tool("get_ticket", {"ticket_id": ticket_id})
    except Exception:
        logger.exception("Ticket lookup failed")
        return reply(
            f"I could not reach the ticket system to check {ticket_id}.",
            "ticket_lookup_unavailable",
            [{"tool": "get_ticket", "ok": False}],
        )

    data, error_message = _tool_result_data(tool_result)
    ticket = data.get("ticket") if data and not error_message else None
    calls = [{"tool": "get_ticket", "ok": error_message is None}]

    raw = getattr(tool_result, "structuredContent", None)

    if not isinstance(raw, dict):
        for item in getattr(tool_result, "content", None) or []:
            try:
                raw = json.loads(getattr(item, "text", ""))
            except (TypeError, ValueError):
                continue
            break

    error_info = raw.get("error") if isinstance(raw, dict) else None
    not_found = isinstance(error_info, dict) and error_info.get("code") == "ticket_not_found"

    if not_found:
        return reply(f"I could not find a ticket {ticket_id}.", "ticket_not_found", calls)

    if not isinstance(ticket, dict):
        return reply(
            f"I could not read ticket {ticket_id} right now.", "ticket_lookup_unavailable", calls
        )

    # Same answer for other people's tickets, so ticket IDs cannot be probed.
    if str(ticket.get("requester_email", "")).strip().lower() != str(
        employee.get("email", signed_in_email)
    ).strip().lower():
        return reply(
            f"I can only show tickets raised by your own account, and I could "
            f"not find {ticket_id} under it.",
            "ticket_not_found",
            calls,
        )

    lines = [
        f"Ticket {ticket_id}: {_single_line(ticket.get('subject'), 120)}",
        f"- Status: {_single_line(ticket.get('status'), 30)}",
        f"- Priority: {_single_line(ticket.get('priority'), 10)}",
        f"- Category: {_single_line(ticket.get('category'), 30)}",
    ]
    if ticket.get("assigned_team"):
        lines.append(f"- Assigned team: {_single_line(ticket['assigned_team'], 60)}")
    if ticket.get("resolution"):
        lines.append(f"- Resolution: {_single_line(ticket['resolution'], 300)}")
    notes = ticket.get("notes")
    if isinstance(notes, list):
        lines.append(f"- Notes on the ticket: {len(notes)}")

    return reply("\n".join(lines), "ticket_status", calls)


_MY_TICKETS_QUESTION = re.compile(
    r"\b(?:my|mt|mine)\s+(?:\w+\s+){0,2}?(?:tickets|tikets|tickts|teckits|tickes)\b"
    r"|\btickets?\s+(?:i|we)\s+(?:raised|opened|created|submitted|have)\b",
    re.IGNORECASE,
)


async def _my_tickets_response(
    message: str, signed_in_email: str | None
) -> dict[str, Any] | None:
    """List the signed-in employee's own tickets through the read-only search_tickets tool."""
    if _TICKET_ID.search(message) or not _MY_TICKETS_QUESTION.search(
        _unquoted_user_text(message)
    ):
        return None

    def reply(text: str, state: str, calls: list[dict[str, Any]] | None = None):
        return {
            "reply": text,
            "pending_action": None,
            "citations": [],
            "tool_calls": calls or [],
            "result": {},
            "state": state,
        }

    employee = None
    if signed_in_email:
        employee, _ = await _lookup_employee(signed_in_email)

    if employee is None:
        return reply(
            "Please sign in with your work email first, so I can show only "
            "the tickets raised by your own account.",
            "ticket_lookup_requires_identity",
        )

    email = str(employee.get("email") or signed_in_email).strip().lower()
    lines = await _recent_ticket_lines(email, limit=10)
    calls = [{"tool": "search_tickets", "ok": bool(lines)}]

    if not lines:
        return reply(
            "I could not find any tickets raised by your account right now.",
            "my_tickets_empty",
            calls,
        )

    return reply(
        "Your tickets:\n" + "\n".join(f"- {line}" for line in lines),
        "my_tickets",
        calls,
    )


@app.get("/api/tickets", summary="List a verified requester's tickets (read-only)")
async def list_tickets(requester_email: str, limit: int = 50):
    """Read-only view of the system of record through the existing search_tickets tool.

    It never creates or changes tickets. The email is a claim, not authentication,
    so it must be a Nakhla employee found in the directory.
    """

    limit = max(1, min(limit, 50))
    employee, error_message = await _lookup_employee(requester_email)

    if error_message or employee is None:
        return JSONResponse(
            status_code=403,
            content={"state": "requester_not_verified", "error": error_message},
        )

    email = requester_email.strip().lower()

    try:
        tool_result = await app.state.mcp.call_tool(
            "search_tickets",
            {"requester_email": email, "limit": limit},
        )
    except Exception:
        logger.exception("Ticket list lookup failed")
        return JSONResponse(
            status_code=502,
            content={"state": "tickets_unavailable"},
        )

    data, error_message = _tool_result_data(tool_result)
    rows = data.get("tickets") if data else None

    if error_message or not isinstance(rows, list):
        return JSONResponse(
            status_code=502,
            content={"state": "tickets_unavailable"},
        )

    tickets = [
        {key: row.get(key) for key in TICKET_LIST_FIELDS}
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("ticket_id"), str)
        and str(row.get("requester_email", "")).strip().lower() == email
    ]
    total = data.get("total_matches")

    return {
        "state": "ok",
        "requester_email": email,
        "tickets": tickets,
        "count": len(tickets),
        "total": total if isinstance(total, int) else len(tickets),
    }


@app.post("/api/identify", summary="Verify a work email against the directory (read-only)")
async def identify(request: IdentifyRequest):
    """Lets a UI greet the employee by name. Identification only: it reads the
    directory through the existing lookup tool and never creates or changes data."""

    employee, error = await _lookup_employee(request.email)

    if employee is None:
        return {"verified": False, "message": error or "Employee not found."}

    return {
        "verified": True,
        "email": employee["email"].strip().lower(),
        "name": employee.get("name"),
        "department": employee.get("department"),
        "title": employee.get("title"),
        "account_status": employee.get("account_status"),
    }


@app.post("/api/memory/forget", summary="Delete this session's saved memory")
async def forget_memory(request: ForgetRequest):
    """Deletes the caller's session memory and, for a directory-verified email,
    that employee's saved facts, notes and past conversations; never touches tickets."""

    store = _memory_store()
    owner: str | None = None

    if request.email and request.email.strip():
        employee, _ = await _lookup_employee(request.email)
        if employee is not None:
            owner = chat_memory.owner_key(employee["email"])

    if store is not None:
        try:
            store.forget(request.session_id, owner)
        except chat_memory.ChatMemoryError:
            logger.exception("Chat memory could not be deleted")
            return {"state": "memory_delete_failed"}

    return {"state": "memory_deleted"}


@app.post("/api/chat", summary="Chat with the IT Helpdesk Agent")
async def chat(request: ChatRequest):
    if request.message == TEST_DRAFT_TRIGGER:
        action = {
            "action_id": f"test-{uuid.uuid4().hex}",
            "summary": (
                "TEST ONLY: pending-action lifecycle check "
                "(not a real ticket)"
            ),
            "test_only": True,
            "payload": {
                "description": (
                    "Test-only placeholder. This is not a real "
                    "support request and must never be submitted "
                    "as a ticket."
                )
            },
        }

        _store_pending_action(request.session_id, action)

        return {
            "reply": (
                "Test-only draft created for checking the approval "
                "flow; no ticket exists."
            ),
            "pending_action": action,
            "citations": [],
            "tool_calls": [],
            "result": {},
            "state": "test_draft_pending",
        }

    input_check = await asyncio.to_thread(
        guardrails_service.guard_input, request.message
    )

    if not input_check.passed:
        return {
            "reply": input_check.text,
            "pending_action": None,
            "citations": [],
            "tool_calls": [],
            "result": {},
            "state": "blocked_by_guardrails",
        }

    try:
        if _asks_to_explain_quoted_ticket_instruction(request.message):
            _clear_ticket_intake(request.session_id)
            reply = (
                "The quoted message is telling its recipient to create a "
                "support ticket right away. Since you asked only for an "
                "explanation, I have not created a ticket."
            )
            _remember_chat_turn(request.session_id, request.message, reply)
            _remember_cited_documents(request.session_id, [])
            return {
                "reply": reply,
                "pending_action": None,
                "citations": [],
                "tool_calls": [],
                "result": {},
            }

        if (
            not _requests_ticket_creation(request.message)
            and _is_service_status_question(request.message)
        ):
            _clear_ticket_intake(request.session_id)
            return await _live_service_status_response(request.message)

        if not _requests_ticket_creation(request.message):
            ticket_reply = await _ticket_lookup_response(
                request.message,
                request.user.email.strip().lower()
                if request.user and request.user.email.strip()
                else None,
            )
            if ticket_reply is None:
                ticket_reply = await _my_tickets_response(
                    request.message,
                    request.user.email.strip().lower()
                    if request.user and request.user.email.strip()
                    else None,
                )
            if ticket_reply is not None:
                _clear_ticket_intake(request.session_id)
                _remember_chat_turn(
                    request.session_id, request.message, ticket_reply["reply"]
                )
                return ticket_reply

        intake = _begin_or_continue_ticket_intake(request)

        verified_lookup: tuple[dict[str, Any] | None, str | None] | None = None
        verified_department: str | None = None

        if intake is not None:
            candidate_email = intake.requester_email or _email_from_message(
                _unquoted_user_text(_ticket_report(intake.messages))
            )
        else:
            candidate_email = (
                request.user.email.strip().lower()
                if request.user and request.user.email.strip()
                else _email_from_message(
                    _unquoted_user_text(request.message)
                )
            )

        # Department exceptions also govern ordinary technical answers, not
        # just ticket drafts. Resolve the submitted work email against the
        # directory before passing a department to the grounded answer path.
        if candidate_email:
            verified_lookup = await _lookup_employee(candidate_email)
            directory_department = (
                verified_lookup[0].get("department")
                if verified_lookup[0] is not None
                else None
            )

            if isinstance(directory_department, str):
                verified_department = directory_department

        if intake is None and verified_department:
            claimed_department = _claimed_department_in_message(request.message)
            if (
                claimed_department
                and claimed_department.casefold()
                != verified_department.casefold()
            ):
                reply = (
                    f"Your message says you're a {claimed_department}, but the "
                    f"verified work account for this chat is associated with "
                    f"{verified_department}. Are you asking on behalf of someone "
                    "in a different department, or should we check a different "
                    "work account? I won't apply a department-specific policy "
                    "until this is clarified."
                )
                _remember_chat_turn(request.session_id, request.message, reply)
                _remember_cited_documents(request.session_id, [])
                return {
                    "reply": reply,
                    "pending_action": None,
                    "citations": [],
                    "tool_calls": [],
                    "result": {},
                    "state": "identity_clarification_required",
                }

        agent_kwargs: dict[str, Any] = {}

        # Long-term memory follows only the signed-in, directory-verified
        # employee, never an address merely mentioned in the message text.
        memory_owner: str | None = None

        if (
            request.user
            and request.user.email.strip()
            and verified_lookup is not None
            and verified_lookup[0] is not None
            and candidate_email == request.user.email.strip().lower()
        ):
            memory_owner = chat_memory.owner_key(candidate_email)

        if (
            intake is None
            and memory_owner
            and verified_lookup is not None
            and verified_lookup[0] is not None
            and _WHO_AM_I.search(_unquoted_user_text(request.message))
        ):
            employee = verified_lookup[0]
            name = _single_line(employee.get("name") or employee.get("full_name"), 80)
            if name:
                reply = (
                    f"You are signed in as {name} ({candidate_email})"
                    + (f", {verified_department}." if verified_department else ".")
                )
                _remember_chat_turn(request.session_id, request.message, reply)
                return {
                    "reply": reply,
                    "pending_action": None,
                    "citations": [],
                    "tool_calls": [{"tool": "lookup_employee", "ok": True}],
                    "result": {},
                    "state": "who_am_i",
                }

        if verified_department:
            agent_kwargs["department"] = verified_department

        # Ticket intake is handled by the application. The model is not asked
        # to troubleshoot again, because its reply would be mixed into the
        # intake prompts (repeated questions, duplicate email requests).
        if intake is not None:
            agent_result = {"reply": "", "citations": [], "tool_calls": []}
        else:
            agent_result = await run_agent(
                message=request.message,
                history=_chat_history(
                    request.session_id,
                    request.message,
                    memory_owner,
                    await _recent_ticket_lines(candidate_email)
                    if memory_owner
                    and candidate_email
                    and _RECURRENCE_HINT.search(request.message)
                    else [],
                ),
                **agent_kwargs,
            )

        citations = (
            []
            if request.message.strip().lower() == "hello"
            else agent_result.get("citations", [])
        )

        if intake is not None:
            draft_result = await _draft_ticket_response(
                request=request,
                intake=intake,
                agent_result=agent_result,
                citations=citations,
                verified_lookup=verified_lookup,
            )
            _remember_intake_turn(
                request.session_id, request.message, draft_result.get("reply")
            )
            return draft_result

        final_reply = agent_result.get(
            "reply",
            agent_result.get("answer", ""),
        )
        _remember_chat_turn(
            request.session_id,
            request.message,
            final_reply,
            citations,
            memory_owner,
        )
        _remember_cited_documents(request.session_id, citations)

        return {
            "reply": final_reply,
            "pending_action": None,
            "citations": citations,
            "tool_calls": agent_result.get("tool_calls", []),
            "result": {},
        }

    except Exception:
        logger.exception("The IT Helpdesk Agent failed")

        return {
            "reply": (
                "Sorry, I could not process your request right now. "
                "Please try again or contact the IT Service Desk."
            ),
            "pending_action": None,
            "citations": [],
            "tool_calls": [],
            "result": {},
        }



def _approval_response(
    reply: str,
    state: str,
    *,
    result: dict[str, Any] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "reply": reply,
        "pending_action": None,
        "citations": [],
        "tool_calls": tool_calls or [],
        "result": result or {},
        "state": state,
    }


def _persist_ticket_reference(
    store: SQLiteTicketMetadataStore | None,
    action_id: str,
    ticket_id: str,
    payload: dict[str, Any],
) -> bool:
    if store is None:
        logger.error(
            "Ticket %s was created, but the local metadata store is unavailable.",
            ticket_id,
        )
        return False

    try:
        store.record_created_ticket(
            idempotency_key=action_id,
            external_ticket_id=ticket_id,
            payload_sha256=payload_digest(payload),
        )
        return True
    except TicketStoreError:
        logger.exception(
            "Ticket %s was created, but its local reference could not be stored.",
            ticket_id,
        )
        return False


EMAIL_FORMAT_ERROR = "That doesn't look like a valid email address."
EMAIL_NOT_VERIFIABLE_ERROR = (
    "That is a valid email address, but it is not a Nakhla work address, "
    "so I can't verify you as an employee."
)
WORK_EMAIL_PROMPT = (
    "Please send your Nakhla work email (name.surname@nakhla.example) "
    "so I can verify the requester."
)


async def _lookup_employee(
    email: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Look up an employee using the read-only MCP tool."""

    if not isinstance(email, str):
        return None, EMAIL_FORMAT_ERROR

    email = email.strip().lower()

    if not _is_valid_email_format(email):
        return None, EMAIL_FORMAT_ERROR

    if not _is_nakhla_email(email):
        # Valid syntax, but the directory only holds Nakhla staff and the
        # ticket schema requires a Nakhla requester: identity is not verified.
        return None, EMAIL_NOT_VERIFIABLE_ERROR

    try:
        tool_result = await app.state.mcp.call_tool(
            "lookup_employee",
            {"email": email},
        )
    except Exception:
        logger.exception("Employee lookup failed")
        return None, "Could not verify the employee at this time."

    data, error_message = _tool_result_data(tool_result)

    if error_message or data is None:
        return None, "Employee verification failed."

    employee = data.get("employee")

    if not isinstance(employee, dict):
        return None, "The employee directory returned invalid data."

    verified_email = employee.get("email")
    if (
        not isinstance(verified_email, str)
        or verified_email.strip().lower() != email
    ):
        return None, "The employee directory returned an inconsistent record."

    if not isinstance(employee.get("department"), str):
        return None, "The employee directory returned invalid department data."

    return employee, None

def _parse_ticket_tool_result(
    tool_result: CallToolResult,
) -> tuple[dict[str, Any] | None, str | None]:

    if tool_result.isError:
        message = next(
            (
                item.text
                for item in tool_result.content
                if getattr(item, "type", None) == "text"
            ),
            "The MCP server reported a tool error.",
        )

        return None, message

    data = tool_result.structuredContent

    if not isinstance(data, dict):
        text = next(
            (
                item.text
                for item in tool_result.content
                if getattr(item, "type", None) == "text"
            ),
            None,
        )

        try:
            data = json.loads(text) if text is not None else None
        except json.JSONDecodeError:
            data = None

    if not isinstance(data, dict):
        return (
            None,
            "The MCP server returned an unreadable ticket result.",
        )

    if data.get("ok") is not True:
        error = data.get("error")
        message = (
            error.get("message")
            if isinstance(error, dict)
            else None
        )

        return (
            None,
            str(
                message
                or "The MCP server did not create the ticket."
            ),
        )

    ticket = data.get("ticket")
    ticket_id = (
        ticket.get("ticket_id")
        if isinstance(ticket, dict)
        else None
    )

    if (
        not isinstance(ticket_id, str)
        or re.fullmatch(r"IT-\d{4}", ticket_id) is None
    ):
        return (
            None,
            "The MCP server did not return a valid ticket ID "
            "(expected IT-####).",
        )

    return {"ticket_id": ticket_id}, None


def _ticket_payload_errors(
    payload: dict[str, Any],
) -> list[str]:

    errors = sorted(
        TICKET_SCHEMA_VALIDATOR.iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )

    return [
        (
            f"{'.'.join(str(part) for part in error.absolute_path) or 'payload'}: "
            f"{error.message}"
        )
        for error in errors
    ]


async def _draft_ticket_response(
    *,
    request: ChatRequest,
    intake: TicketIntake,
    agent_result: dict[str, Any],
    citations: list[dict[str, Any]],
    verified_lookup: tuple[dict[str, Any] | None, str | None] | None = None,
) -> dict[str, Any]:
    report = _ticket_report(intake.messages)
    facts = _unquoted_user_text(report)
    reply = str(
        agent_result.get("reply")
        or agent_result.get("answer")
        or ""
    ).strip()
    tool_calls = agent_result.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        tool_calls = []
    tool_calls = list(tool_calls)
    service = _ticket_service(facts)

    try:
        status_result = await app.state.mcp.call_tool(
            "get_service_status",
            {},
        )
    except Exception:
        logger.exception("Service status lookup failed")
        status_data = None
        status_error = "Could not check the live service status."
    else:
        status_data, status_error = _tool_result_data(status_result)

    tool_calls.append(
        {
            "tool": "get_service_status",
            "ok": status_error is None,
        }
    )

    status_page: dict[str, Any] | None = None

    if status_error is None and status_data is not None:
        candidate_status = status_data.get("status")

        if (
            isinstance(candidate_status, dict)
            and isinstance(candidate_status.get("services"), list)
            and isinstance(candidate_status.get("incidents"), list)
            and all(
                isinstance(service_info, dict)
                and isinstance(service_info.get("id"), str)
                and isinstance(service_info.get("status"), str)
                for service_info in candidate_status["services"]
            )
            and all(
                isinstance(incident_info, dict)
                and isinstance(incident_info.get("id"), str)
                and isinstance(incident_info.get("service"), str)
                and isinstance(incident_info.get("status"), str)
                and (
                    "error_codes" not in incident_info
                    or isinstance(incident_info.get("error_codes"), list)
                )
                for incident_info in candidate_status["incidents"]
            )
        ):
            status_page = status_data
        else:
            status_error = "The service status tool returned malformed data."
            tool_calls[-1]["ok"] = False

    if status_error:
        return {
            "reply": (
                f"{reply}\n\n{status_error} I will not prepare a ticket "
                "draft until the status check succeeds."
            ).strip(),
            "pending_action": None,
            "citations": citations,
            "tool_calls": tool_calls,
            "result": {},
            "state": "status_check_failed",
        }

    candidate_email = intake.requester_email
    if not candidate_email:
        candidate_email = _email_from_message(facts)

    employee: dict[str, Any] | None = None
    employee_error: str | None = None

    if candidate_email:
        if verified_lookup is not None:
            employee, employee_error = verified_lookup
        else:
            employee, employee_error = await _lookup_employee(candidate_email)
        tool_calls.append(
            {
                "tool": "lookup_employee",
                "ok": employee is not None,
            }
        )

    if employee_error:
        return {
            "reply": (
                f"{reply}\n\n{employee_error} {WORK_EMAIL_PROMPT}"
            ).strip(),
            "pending_action": None,
            "citations": citations,
            "tool_calls": tool_calls,
            "result": {},
            "state": "requester_unverified",
        }

    if employee is not None:
        department_value = employee.get("department")
        departments = TICKET_SCHEMA.get("$defs", {}).get(
            "department", {}
        ).get("enum", [])

        if (
            not isinstance(department_value, str)
            or department_value not in departments
        ):
            return {
                "reply": (
                    f"{reply}\n\nThe employee directory returned an "
                    "unsupported department. No draft was created."
                ).strip(),
                "pending_action": None,
                "citations": citations,
                "tool_calls": tool_calls,
                "result": {},
                "state": "requester_unverified",
            }

        if _requester_identity_conflicts(intake, request, employee):
            return {
                "reply": (
                    "The requester details in this conversation do not match "
                    "the verified employee record. No draft was prepared. "
                    f"{WORK_EMAIL_PROMPT}"
                ),
                "pending_action": None,
                "citations": citations,
                "tool_calls": tool_calls,
                "result": {},
                "state": "requester_identity_mismatch",
            }

        if (
            department_value == "Contractors"
            and service == "vpn"
        ):
            _clear_ticket_intake(request.session_id)
            return {
                "reply": (
                    f"{reply}\n\nContractors do not receive VPN access. "
                    "Use the Nakhla Workspace VDI portal instead; no "
                    "ticket has been prepared."
                ).strip(),
                "pending_action": None,
                "citations": citations,
                "tool_calls": tool_calls,
                "result": {},
                "state": "ticket_not_applicable",
            }

    incident = (
        _active_matching_incident(status_page, facts, service)
        if status_page is not None and employee is not None
        else None
    )

    if incident is not None:
        incident_id = str(incident.get("id", "active incident"))
        incident_details = [
            f"A matching active incident ({incident_id}) is already being handled."
        ]

        workaround = incident.get("workaround")
        instructions = incident.get("instructions")
        if isinstance(workaround, str) and workaround.strip():
            incident_details.append(f"Workaround: {workaround.strip()}")
        if isinstance(instructions, str) and instructions.strip():
            incident_details.append(instructions.strip())

        _clear_ticket_intake(request.session_id)
        return {
            "reply": "\n\n".join(
                ([reply] if reply else []) + incident_details
            ),
            "pending_action": None,
            "citations": citations,
            "tool_calls": tool_calls,
            "result": {},
            "state": "active_incident",
        }

    missing: list[str] = []
    if employee is None:
        missing.append(
            "your Nakhla work email (I will check it against the employee directory)"
        )

    description = report.strip()
    subject = _ticket_subject(facts)
    attempted_steps = _attempted_steps(intake.messages)
    affected_users = _extract_affected_users(facts)
    kb_refs = _extract_kb_refs(intake.messages)

    for ref in _followed_guidance_refs(request.session_id, intake.messages):
        if ref not in kb_refs:
            kb_refs.append(ref)

    if not attempted_steps and employee is not None:
        direct = _kb_direct_escalation_step(
            facts,
            str(employee.get("department") or ""),
            service,
            status_checked=status_page is not None,
        )

        if direct is not None:
            attempted_steps = [direct[0]]
            if direct[1] not in kb_refs:
                kb_refs.append(direct[1])

    if len(description) < 10 or not subject:
        missing.append("a clear description of the problem")
    if len(description) > 4000:
        missing.append(
            "a shorter problem description (the ticket limit is 4,000 characters)"
        )
    if not attempted_steps:
        missing.append(
            "at least one troubleshooting step you actually tried"
        )
    if any(len(step) > 300 for step in attempted_steps):
        missing.append(
            "a shorter description of each troubleshooting step (300 characters maximum)"
        )
    if len(attempted_steps) > 20:
        missing.append(
            "no more than 20 troubleshooting steps"
        )
    if len(kb_refs) > 10:
        missing.append(
            "no more than 10 knowledge-base references that you actually followed"
        )
    if affected_users is not None and not 1 <= affected_users <= 1000:
        missing.append(
            "a valid affected-user count between 1 and 1,000"
        )

    department: str | None = None
    category: str | None = None
    priority: str | None = None
    if employee is not None:
        department_value = employee.get("department")
        departments = TICKET_SCHEMA.get("$defs", {}).get(
            "department", {}
        ).get("enum", [])

        if (
            not isinstance(department_value, str)
            or department_value not in departments
        ):
            employee_error = (
                "The employee directory returned an unsupported department."
            )
        else:
            department = department_value
            category = _ticket_category(facts, department)

            priority = _ticket_priority(
                facts,
                department,
                affected_users,
            )
            if priority is None:
                missing.append(
                    "the impact required by KB-015: how many people are "
                    "affected, and whether work is completely blocked with "
                    "no workaround or a workaround exists"
                )
            for reference in _classification_kb_refs(
                category,
                priority,
                department,
                facts,
            ):
                if reference not in kb_refs:
                    kb_refs.append(reference)

    if len(kb_refs) > 10:
        missing.append(
            "no more than 10 relevant knowledge-base references"
        )

    if employee_error:
        return {
            "reply": (
                f"{reply}\n\n{employee_error} No draft was created."
            ).strip(),
            "pending_action": None,
            "citations": citations,
            "tool_calls": tool_calls,
            "result": {},
            "state": "requester_unverified",
        }

    if missing:
        if employee is None:
            # Verify the requester first; ask for the rest one step later.
            missing = missing[:1]
        questions = "\n".join(f"- {item}" for item in dict.fromkeys(missing))
        return {
            "reply": (
                f"{reply}\n\nTo prepare a valid ticket draft, please provide:\n"
                f"{questions}"
            ).strip(),
            "pending_action": None,
            "citations": citations,
            "tool_calls": tool_calls,
            "result": {},
            "state": "ticket_info_required",
        }

    assert employee is not None
    assert department is not None
    assert category is not None
    assert priority is not None

    requester_email = str(employee["email"]).strip().lower()
    payload: dict[str, Any] = {
        "subject": subject,
        "description": description,
        "category": category,
        "priority": priority,
        "requester_email": requester_email,
        "department": department,
        "attempted_steps": attempted_steps,
        "kb_refs": kb_refs,
    }

    logger.info(
        "Ticket draft staged: category=%s priority=%s kb_refs=%s; approval_required=true",
        category,
        priority,
        sorted(kb_refs),
    )

    if affected_users is not None:
        payload["affected_users"] = affected_users

    payload_errors = _ticket_payload_errors(payload)
    if payload_errors:
        logger.error(
            "Generated ticket draft failed schema validation: %s",
            "; ".join(payload_errors),
        )
        return {
            "reply": (
                f"{reply}\n\nI could not validate the proposed ticket "
                "against the ticket schema, so no draft was registered. "
                "Please clarify the problem and the steps you tried."
            ).strip(),
            "pending_action": None,
            "citations": citations,
            "tool_calls": tool_calls,
            "result": {},
            "state": "invalid_ticket_draft",
        }

    action = {
        "action_id": uuid.uuid4().hex,
        "type": "create_ticket",
        "summary": f"Create support ticket: {subject}",
        "payload": payload,
    }
    _store_pending_action(request.session_id, action)
    _clear_ticket_intake(request.session_id)

    step_summary = "; ".join(attempted_steps)
    draft_summary = (
        "Ticket draft prepared; no ticket has been created.\n"
        f"Subject: {subject}\n"
        f"Category: {category}\n"
        f"Priority: {priority}\n"
        + f"Requester: {requester_email} ({department})\n"
        f"Attempted steps: {step_summary}\n"
        f"KB references: {', '.join(payload['kb_refs']) or 'none'}\n"
        "Review the complete draft and explicitly approve it to submit."
    )

    return {
        "reply": "\n\n".join(([reply] if reply else []) + [draft_summary]),
        "pending_action": action,
        "citations": citations,
        "tool_calls": tool_calls,
        "result": {},
        "state": "ticket_draft_pending",
    }


@app.post(
    "/api/approve",
    summary="Approve or reject a registered action",
)
async def approve(
    request: ApproveRequest,
    http_request: Request,
):
    key = (
        request.session_id,
        request.action_id,
    )

    with http_request.app.state.pending_actions_lock:
        record = http_request.app.state.pending_actions.get(key)

        if record is None:
            record = _restore_approval_state(
                http_request.app.state,
                request.session_id,
                request.action_id,
            )
            if record is not None:
                http_request.app.state.pending_actions[key] = record

        if record is None:
            return _approval_response(
                "No matching pending action was found for this "
                "session. Nothing was approved or changed.",
                "no_pending_action",
            )

        if record.decision is not None:
            if record.decision != request.approved:
                return _approval_response(
                    "This action already has a different recorded "
                    "decision. No change was made.",
                    "decision_already_recorded",
                )

            if not request.approved:
                return _approval_response(
                    "This action was already rejected. "
                    "No action was taken.",
                    "already_rejected",
                )

            if record.action.get("test_only"):
                return _approval_response(
                    "Test-only approval recorded. "
                    "No ticket was created.",
                    "already_approved",
                )

            if record.write_status == "succeeded":
                if (
                    not record.local_recorded
                    and record.ticket_id
                    and isinstance(record.action.get("payload"), dict)
                ):
                    record.local_recorded = _persist_ticket_reference(
                        getattr(
                            http_request.app.state,
                            "ticket_store",
                            None,
                        ),
                        request.action_id,
                        record.ticket_id,
                        record.action["payload"],
                    )
                    if record.local_recorded:
                        _persist_approval_state(
                            http_request.app.state,
                            request.session_id,
                            record,
                        )
                if not record.local_recorded:
                    return _approval_response(
                        (
                            f"Ticket {record.ticket_id} was already created, "
                            "but its local metadata reference could not be "
                            "saved. No duplicate ticket was created."
                        ),
                        "ticket_created_local_persistence_failed",
                        result=(
                            {"ticket_id": record.ticket_id}
                            if record.ticket_id
                            else {}
                        ),
                    )

                return _approval_response(
                    (
                        f"Ticket {record.ticket_id} was already "
                        "created for this approved action."
                    ),
                    "already_approved",
                    result=(
                        {"ticket_id": record.ticket_id}
                        if record.ticket_id
                        else {}
                    ),
                )

            if record.write_status == "in_progress":
                return _approval_response(
                    "Ticket creation is already in progress "
                    "for this action.",
                    "creation_in_progress",
                )

        elif not request.approved:
            record.decision = False
            record.write_status = "not_started"
            _persist_approval_state(
                http_request.app.state, request.session_id, record
            )

            return _approval_response(
                "Pending action rejected. No action was taken.",
                "rejected",
            )

        else:
            record.decision = True

        action = record.action

        if action.get("test_only"):
            return _approval_response(
                "Test-only approval recorded. "
                "No ticket was created.",
                "approved",
            )

        if action.get("action_id") != request.action_id:
            record.write_status = "failed"

            return _approval_response(
                "The registered action is inconsistent. "
                "No ticket was created.",
                "invalid_pending_action",
            )

        if action.get("type") != "create_ticket":
            record.write_status = "failed"

            return _approval_response(
                "This pending action is not registered for "
                "ticket creation. No ticket was created.",
                "unsupported_action",
            )

        payload = action.get("payload")

        if not isinstance(payload, dict):
            record.write_status = "failed"

            return _approval_response(
                "The registered ticket payload is invalid. "
                "No ticket was created.",
                "invalid_pending_action",
            )

        payload_errors = _ticket_payload_errors(payload)

        if payload_errors:
            record.write_status = "failed"

            return _approval_response(
                (
                    "The registered ticket payload does not match "
                    "the ticket schema. "
                    + "; ".join(payload_errors)
                    + " No ticket was created."
                ),
                "invalid_pending_action",
            )

        authorization = ApprovedTicketAuthorization(
            action_id=request.action_id,
            payload=deepcopy(payload),
        )

        record.write_status = "in_progress"
        _persist_approval_state(
            http_request.app.state, request.session_id, record
        )
    try:
        logger.info(
            "Explicit ticket approval accepted; invoking create_ticket for category=%s priority=%s kb_refs=%s",
            payload.get("category"),
            payload.get("priority"),
            sorted(payload.get("kb_refs", [])),
        )
        tool_result = (
            await http_request.app.state.mcp.create_approved_ticket(
                authorization
            )
        )

        ticket_result, error_message = (
            _parse_ticket_tool_result(tool_result)
        )

    except Exception as exc:
        logger.exception(
            "Approved ticket creation failed for action %s",
            request.action_id,
        )

        ticket_result = None
        error_message = f"{type(exc).__name__}: {exc}"

    local_recorded = False
    if ticket_result is not None:
        local_recorded = _persist_ticket_reference(
            getattr(http_request.app.state, "ticket_store", None),
            request.action_id,
            ticket_result["ticket_id"],
            payload,
        )

    with http_request.app.state.pending_actions_lock:
        current_record = (
            http_request.app.state.pending_actions.get(key)
        )

        if current_record is None:
            return _approval_response(
                "The pending action was removed while ticket "
                "creation ran; its result could not be recorded.",
                "result_not_recorded",
            )

        if ticket_result is None:
            current_record.write_status = "failed"
            _persist_approval_state(
                http_request.app.state, request.session_id, current_record
            )
            return _approval_response(
                (
                    f"Ticket creation was not confirmed: "
                    f"{error_message} "
                    "You may retry this same approved action."
                ),
                "ticket_creation_failed",
            )

        current_record.write_status = "succeeded"
        current_record.ticket_id = ticket_result["ticket_id"]
        current_record.local_recorded = local_recorded
        _persist_approval_state(
            http_request.app.state, request.session_id, current_record
        )
    created_call = [{"tool": "create_ticket", "ok": True}]

    if not local_recorded:
        return _approval_response(
            (
                f"Ticket {ticket_result['ticket_id']} was created, but its "
                "local metadata reference could not be saved. "
                "No duplicate ticket will be created for this action."
            ),
            "ticket_created_local_persistence_failed",
            result=ticket_result,
            tool_calls=created_call,
        )

    logger.info(
        "Approved ticket creation completed: system_recorded=true local_reference_recorded=true",
    )

    return _approval_response(
        f"Ticket {ticket_result['ticket_id']} was created.",
        "ticket_created",
        result=ticket_result,
        tool_calls=created_call,
    )
