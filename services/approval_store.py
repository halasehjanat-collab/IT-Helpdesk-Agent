"""Minimal durable approval state so a pending ticket draft survives an app restart.

Stored in the app metadata database as an additive table (the existing ticket_references table is
untouched). Only the validated draft the user was shown, its digest, the decision and the write
status are kept. The session ID is stored only as a SHA-256 hash, so a restored action can only be
approved by the session that owns it. Test-only actions are never persisted. Once a ticket has been
created and recorded locally, the draft body is scrubbed and only the digest remains.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from services.local_ticket_store import payload_digest


class ApprovalStoreError(RuntimeError):
    """Approval state could not be stored or read."""


def session_hash(session_id: str) -> str:
    return sha256(session_id.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SQLiteApprovalStore:
    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.database_path, timeout=10, isolation_level=None)
            connection.row_factory = sqlite3.Row
            return connection
        except (OSError, sqlite3.Error) as exc:
            raise ApprovalStoreError("Could not open the approval store.") from exc

    def initialize(self) -> None:
        try:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS approval_actions (
                        session_hash TEXT NOT NULL,
                        action_id TEXT NOT NULL,
                        action_json TEXT NOT NULL,
                        payload_sha256 TEXT NOT NULL,
                        decision INTEGER CHECK (decision IN (0, 1)),
                        write_status TEXT NOT NULL,
                        ticket_id TEXT,
                        local_recorded INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (session_hash, action_id)
                    )
                    """
                )
        except sqlite3.Error as exc:
            raise ApprovalStoreError("Could not initialize the approval store.") from exc

    def save(
        self,
        session_id: str,
        action: dict[str, Any],
        *,
        decision: bool | None,
        write_status: str,
        ticket_id: str | None,
        local_recorded: bool,
    ) -> None:
        """Insert or update state. The stored draft and its digest are never changed by an update,
        and an existing decision can never be overwritten."""
        if action.get("test_only"):
            return
        action_id = action.get("action_id")
        payload = action.get("payload")
        if not isinstance(action_id, str) or not isinstance(payload, dict):
            raise ApprovalStoreError("Only complete ticket drafts can be stored.")
        digest = payload_digest(payload)
        scrubbed = write_status == "succeeded" and local_recorded
        stored_action = (
            {"type": action.get("type"), "action_id": action_id}
            if scrubbed
            else {key: action[key] for key in ("type", "action_id", "summary", "payload") if key in action}
        )
        now = _now()
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.execute(
                        """
                        INSERT INTO approval_actions (session_hash, action_id, action_json, payload_sha256,
                            decision, write_status, ticket_id, local_recorded, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (session_hash, action_id) DO UPDATE SET
                            decision = COALESCE(approval_actions.decision, excluded.decision),
                            write_status = excluded.write_status,
                            ticket_id = COALESCE(excluded.ticket_id, approval_actions.ticket_id),
                            local_recorded = MAX(approval_actions.local_recorded, excluded.local_recorded),
                            action_json = CASE WHEN ? THEN excluded.action_json ELSE approval_actions.action_json END,
                            updated_at = excluded.updated_at
                        """,
                        (
                            session_hash(session_id), action_id, json.dumps(stored_action), digest,
                            None if decision is None else int(decision), write_status, ticket_id,
                            int(local_recorded), now, now, int(scrubbed),
                        ),
                    )
                    connection.execute("COMMIT")
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
        except sqlite3.Error as exc:
            raise ApprovalStoreError("Could not save approval state.") from exc

    def load(self, session_id: str, action_id: str) -> dict[str, Any] | None:
        """Return the persisted record for this session, or None if missing or tampered with."""
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT action_json, payload_sha256, decision, write_status, ticket_id, local_recorded "
                    "FROM approval_actions WHERE session_hash = ? AND action_id = ?",
                    (session_hash(session_id), action_id),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ApprovalStoreError("Could not read approval state.") from exc
        if row is None:
            return None
        try:
            action = json.loads(row["action_json"])
        except ValueError:
            return None
        local_recorded = bool(row["local_recorded"])
        scrubbed = row["write_status"] == "succeeded" and local_recorded
        if not scrubbed:
            payload = action.get("payload") if isinstance(action, dict) else None
            if not isinstance(payload, dict) or payload_digest(payload) != row["payload_sha256"]:
                return None
        write_status = row["write_status"]
        if write_status == "in_progress":
            # The process stopped mid-write; the idempotency key makes a retry safe.
            write_status = "failed"
        return {
            "action": action,
            "decision": None if row["decision"] is None else bool(row["decision"]),
            "write_status": write_status,
            "ticket_id": row["ticket_id"],
            "local_recorded": local_recorded,
        }
