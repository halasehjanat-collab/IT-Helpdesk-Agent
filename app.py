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

from fastapi import FastAPI, Request
from pydantic import BaseModel
from jsonschema import Draft202012Validator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Tool

READ_ONLY_MCP_TOOL_NAMES = frozenset(
    {"lookup_employee", "get_service_status", "search_tickets", "get_ticket"}
)
TEST_DRAFT_TRIGGER = "/test-draft"
logger = logging.getLogger(__name__)

TICKET_SCHEMA_PATH = (
    Path(__file__).resolve().parent / "dataset" / "schemas" / "ticket.schema.json"
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
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> CallToolResult:
        if name not in self._tool_names:
            raise ValueError(f"MCP tool is not available for chat: {name}")
        return await self._session.call_tool(name, arguments)

    async def create_approved_ticket(
        self, authorization: ApprovedTicketAuthorization
    ) -> CallToolResult:
        if not isinstance(authorization, ApprovedTicketAuthorization):
            raise TypeError("Ticket creation requires an approval authorization.")
        arguments = deepcopy(authorization.payload)
        arguments["idempotency_key"] = authorization.action_id
        return await self._session.call_tool("create_ticket", arguments)


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = Path(__file__).resolve().parent
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
            app.state.mcp = ReadOnlyMCPClient(session, read_only_tools)
            yield


app = FastAPI(lifespan=lifespan)
app.state.pending_actions = {}
app.state.pending_actions_lock = Lock()


@dataclass
class PendingActionRecord:
    action: dict[str, Any]
    decision: bool | None = None
    write_status: str = "not_started"
    ticket_id: str | None = None


def _store_pending_action(session_id: str, action: dict[str, Any]) -> None:
    action_id = action.get("action_id")
    if not session_id.strip() or not isinstance(action_id, str) or not action_id.strip():
        raise ValueError("A pending action requires a session ID and action ID.")

    key = (session_id, action_id)
    with app.state.pending_actions_lock:
        if key in app.state.pending_actions:
            raise ValueError("This pending action already exists for the session.")
        app.state.pending_actions[key] = PendingActionRecord(action=deepcopy(action))


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


@app.post("/api/chat", summary="Temporary chat placeholder")
async def chat(request: ChatRequest):
    if request.message == TEST_DRAFT_TRIGGER:
        action = {
            "action_id": f"test-{uuid.uuid4().hex}",
            "summary": "TEST ONLY: pending-action lifecycle check (not a real ticket)",
            "test_only": True,
            "payload": {
                "description": (
                    "Test-only placeholder. This is not a real support request "
                    "and must never be submitted as a ticket."
                )
            },
        }
        _store_pending_action(request.session_id, action)
        return {
            "reply": "Test-only draft created for checking the approval flow; no ticket exists.",
            "pending_action": action,
            "citations": [],
            "tool_calls": [],
            "result": {},
            "state": "test_draft_pending",
        }

    return {
        "reply": "Chat is a temporary placeholder and is not implemented yet.",
        "pending_action": None,
        "citations": [],
        "tool_calls": [],
        "result": {},
    }


def _approval_response(
    reply: str, state: str, *, result: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "reply": reply,
        "pending_action": None,
        "citations": [],
        "tool_calls": [],
        "result": result or {},
        "state": state,
    }


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
        return None, "The MCP server returned an unreadable ticket result."
    if data.get("ok") is not True:
        error = data.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        return None, str(message or "The MCP server did not create the ticket.")
    ticket = data.get("ticket")
    ticket_id = ticket.get("ticket_id") if isinstance(ticket, dict) else None
    if not isinstance(ticket_id, str) or re.fullmatch(r"IT-\d{4}", ticket_id) is None:
        return None, "The MCP server did not return a valid ticket ID (expected IT-####)."
    return {"ticket_id": ticket_id}, None


def _ticket_payload_errors(payload: dict[str, Any]) -> list[str]:
    errors = sorted(
        TICKET_SCHEMA_VALIDATOR.iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    return [
        f"{'.'.join(str(part) for part in error.absolute_path) or 'payload'}: "
        f"{error.message}"
        for error in errors
    ]


@app.post("/api/approve", summary="Approve or reject a registered action")
async def approve(request: ApproveRequest, http_request: Request):
    key = (request.session_id, request.action_id)
    with http_request.app.state.pending_actions_lock:
        record = http_request.app.state.pending_actions.get(key)
        if record is None:
            return _approval_response(
                "No matching pending action was found for this session. Nothing was approved or changed.",
                "no_pending_action",
            )

        if record.decision is not None:
            if record.decision != request.approved:
                return _approval_response(
                    "This action already has a different recorded decision. No change was made.",
                    "decision_already_recorded",
                )
            if not request.approved:
                return _approval_response(
                    "This action was already rejected. No action was taken.",
                    "already_rejected",
                )
            if record.action.get("test_only"):
                return _approval_response(
                    "Test-only approval recorded. No ticket was created.",
                    "already_approved",
                )
            if record.write_status == "succeeded":
                return _approval_response(
                    f"Ticket {record.ticket_id} was already created for this approved action.",
                    "already_approved",
                    result={"ticket_id": record.ticket_id} if record.ticket_id else {},
                )
            if record.write_status == "in_progress":
                return _approval_response(
                    "Ticket creation is already in progress for this action.",
                    "creation_in_progress",
                )
        elif not request.approved:
            record.decision = False
            record.write_status = "not_started"
            return _approval_response(
                "Pending action rejected. No action was taken.",
                "rejected",
            )
        else:
            record.decision = True

        action = record.action
        if action.get("test_only"):
            return _approval_response(
                "Test-only approval recorded. No ticket was created.",
                "approved",
            )
        if action.get("action_id") != request.action_id:
            record.write_status = "failed"
            return _approval_response(
                "The registered action is inconsistent. No ticket was created.",
                "invalid_pending_action",
            )
        if action.get("type") != "create_ticket":
            record.write_status = "failed"
            return _approval_response(
                "This pending action is not registered for ticket creation. No ticket was created.",
                "unsupported_action",
            )

        payload = action.get("payload")
        if not isinstance(payload, dict):
            record.write_status = "failed"
            return _approval_response(
                "The registered ticket payload is invalid. No ticket was created.",
                "invalid_pending_action",
            )
        payload_errors = _ticket_payload_errors(payload)
        if payload_errors:
            record.write_status = "failed"
            return _approval_response(
                "The registered ticket payload does not match the ticket schema. "
                + "; ".join(payload_errors)
                + " No ticket was created.",
                "invalid_pending_action",
            )

        authorization = ApprovedTicketAuthorization(
            action_id=request.action_id,
            payload=deepcopy(payload),
        )
        record.write_status = "in_progress"

    try:
        tool_result = await http_request.app.state.mcp.create_approved_ticket(
            authorization
        )
        ticket_result, error_message = _parse_ticket_tool_result(tool_result)
    except Exception as exc:
        logger.exception("Approved ticket creation failed for action %s", request.action_id)
        ticket_result = None
        error_message = f"{type(exc).__name__}: {exc}"

    with http_request.app.state.pending_actions_lock:
        current_record = http_request.app.state.pending_actions.get(key)
        if current_record is None:
            return _approval_response(
                "The pending action was removed while ticket creation ran; its result could not be recorded.",
                "result_not_recorded",
            )
        if ticket_result is None:
            current_record.write_status = "failed"
            return _approval_response(
                f"Ticket creation was not confirmed: {error_message} "
                "You may retry this same approved action.",
                "ticket_creation_failed",
            )
        current_record.write_status = "succeeded"
        current_record.ticket_id = ticket_result["ticket_id"]

    return _approval_response(
        f"Ticket {ticket_result['ticket_id']} was created.",
        "ticket_created",
        result=ticket_result,
    )
