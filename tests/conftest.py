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


@pytest.fixture(autouse=True)
def _block_real_mcp(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("Tests must never start the real MCP subprocess.")

    monkeypatch.setattr(app_module, "stdio_client", _forbidden)
    monkeypatch.setattr(
        app_module.ReadOnlyMCPClient, "create_approved_ticket", _forbidden
    )


@pytest.fixture
def fake_mcp():
    fake = FakeMCP()
    previous = getattr(app_module.app.state, "mcp", None)
    app_module.app.state.mcp = fake
    yield fake
    app_module.app.state.mcp = previous


@pytest.fixture
def client(fake_mcp):
    # No `with`: the lifespan (real MCP subprocess) is intentionally not run.
    app_module.app.state.pending_actions.clear()
    yield TestClient(app_module.app)
    app_module.app.state.pending_actions.clear()


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
