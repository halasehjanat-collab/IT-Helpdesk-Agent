import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from mcp.types import CallToolResult

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app as app_module  # noqa: E402
from services.approval_store import SQLiteApprovalStore  # noqa: E402
from services.local_ticket_store import SQLiteTicketMetadataStore  # noqa: E402

VALID_PAYLOAD = {
    "subject": "VPN error NX-781 on login",
    "description": "NakhlaConnect fails with NX-781 since this morning.",
    "category": "network_access",
    "priority": "P3",
    "requester_email": "dana.levi@nakhla.example",
    "department": "Engineering",
    "attempted_steps": ["Restarted the VPN client"],
    "kb_refs": ["KB-002"],
}


def ticket_result(ticket_id: str = "IT-2001") -> CallToolResult:
    return CallToolResult(
        content=[],
        structuredContent={"ok": True, "ticket": {"ticket_id": ticket_id}},
        isError=False,
    )


class FakeMCP:
    """Stand-in for app.state.mcp; the writer is an AsyncMock."""

    def __init__(self):
        self.create_approved_ticket = AsyncMock(return_value=ticket_result())
        self.call_tool = AsyncMock(side_effect=self._read_tool)

    async def _read_tool(self, name, arguments=None):
        if name == "lookup_employee":
            email = (arguments or {}).get(
                "email",
                "dana.levi@nakhla.example",
            )
            return CallToolResult(
                content=[],
                structuredContent={
                    "ok": True,
                    "employee": {
                        "email": email,
                        "department": "Engineering",
                    },
                },
                isError=False,
            )

        if name == "get_service_status":
            return CallToolResult(
                content=[],
                structuredContent={
                    "ok": True,
                    "status": {
                        "services": [],
                        "incidents": [],
                    },
                },
                isError=False,
            )

        raise AssertionError(f"Unexpected read-only MCP tool call: {name}")


@pytest.fixture(autouse=True)
def _block_real_mcp(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("Tests must never start the real MCP subprocess.")

    monkeypatch.setattr(app_module, "stdio_client", _forbidden)
    monkeypatch.setattr(
        app_module.ReadOnlyMCPClient, "create_approved_ticket", _forbidden
    )


@pytest.fixture
def fake_mcp(tmp_path):
    fake = FakeMCP()
    previous = getattr(app_module.app.state, "mcp", None)
    previous_store = getattr(app_module.app.state, "ticket_store", None)
    previous_approvals = getattr(app_module.app.state, "approval_store", None)
    app_module.app.state.mcp = fake
    store = SQLiteTicketMetadataStore(tmp_path / "ticket_metadata.sqlite3")
    store.initialize()
    app_module.app.state.ticket_store = store
    approvals = SQLiteApprovalStore(tmp_path / "ticket_metadata.sqlite3")
    approvals.initialize()
    app_module.app.state.approval_store = approvals
    yield fake
    app_module.app.state.mcp = previous
    app_module.app.state.ticket_store = previous_store
    app_module.app.state.approval_store = previous_approvals


@pytest.fixture
def ticket_api(tmp_path, monkeypatch):
    """Factory for TestClients on the simulated helpdesk API, each backed by the same temp database.

    Calling it again simulates an API restart: the module-level state is dropped and rebuilt
    from the SQLite file. The default development database is never used.
    """
    import services.helpdesk_api as api_module

    db_path = tmp_path / "tickets.sqlite3"
    monkeypatch.setenv("HELPDESK_TICKET_DB_PATH", str(db_path))
    monkeypatch.delenv("HELPDESK_ALLOW_DESTRUCTIVE_RESET", raising=False)
    monkeypatch.setattr(api_module, "TICKET_DB", None)

    def _start():
        api_module.TICKET_DB = None
        api_module.reset_state()
        return TestClient(api_module.app, headers={"X-API-Key": api_module.API_KEY})

    _start.db_path = db_path
    _start.module = api_module
    yield _start
    api_module.TICKET_DB = None


@pytest.fixture
def client(fake_mcp):
    # No `with`: the lifespan (real MCP subprocess) is intentionally not run.
    app_module.app.state.pending_actions.clear()
    app_module.app.state.ticket_intakes.clear()
    yield TestClient(app_module.app)
    app_module.app.state.pending_actions.clear()
    app_module.app.state.ticket_intakes.clear()


@pytest.fixture
def seed_ticket_action():
    def _seed(session_id="s1", action_id="act-1", **overrides):
        action = {
            "action_id": action_id,
            "type": "create_ticket",
            "summary": "Create ticket",
            "payload": dict(VALID_PAYLOAD),
        }
        action.update(overrides)
        app_module._store_pending_action(session_id, action)
        return action

    return _seed
