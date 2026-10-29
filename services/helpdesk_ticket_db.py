"""Durable SQLite storage for tickets created through the simulated helpdesk API.

This database is owned by the simulated helpdesk (the system of record in this kit). It stores
complete ticket records, notes and idempotency records. The historical JSON archive is NOT
copied here: it stays a read-only seed. Notes added to archived tickets are stored here and
overlaid on read. The app's separate metadata database (services/local_ticket_store.py) is a
different file with a different purpose.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

FIRST_CREATED_TICKET_NUMBER = 2001
DEFAULT_DB_NAME = "helpdesk_tickets.sqlite3"


class TicketDatabaseError(RuntimeError):
    """The ticket database could not be opened, read or written."""


class IdempotencyConflict(TicketDatabaseError):
    """An Idempotency-Key was reused with a different request body."""


def configured_ticket_db_path(kit_dir: Path, data_dir: Path) -> Path:
    """HELPDESK_TICKET_DB_PATH (absolute, or relative to the kit folder) or <data dir>/helpdesk_tickets.sqlite3."""
    configured = (os.getenv("HELPDESK_TICKET_DB_PATH") or "").strip()
    if not configured:
        return (data_dir / DEFAULT_DB_NAME).resolve()
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = kit_dir / path
    return path.resolve()


_TICKET_COLUMNS = (
    "ticket_id, subject, description, category, priority, status, requester_email, department, "
    "affected_users, assigned_team, attempted_steps_json, kb_refs_json, resolution, created_at, "
    "resolved_at, warnings_json"
)

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS tickets (
        ticket_number INTEGER PRIMARY KEY,
        ticket_id TEXT NOT NULL UNIQUE,
        subject TEXT NOT NULL,
        description TEXT NOT NULL,
        category TEXT NOT NULL,
        priority TEXT NOT NULL,
        status TEXT NOT NULL,
        requester_email TEXT NOT NULL,
        department TEXT NOT NULL,
        affected_users INTEGER NOT NULL,
        assigned_team TEXT NOT NULL,
        attempted_steps_json TEXT NOT NULL,
        kb_refs_json TEXT NOT NULL,
        resolution TEXT,
        created_at TEXT NOT NULL,
        resolved_at TEXT,
        warnings_json TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tickets_category ON tickets (category)",
    "CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets (status)",
    "CREATE INDEX IF NOT EXISTS idx_tickets_department ON tickets (department)",
    "CREATE INDEX IF NOT EXISTS idx_tickets_requester ON tickets (requester_email)",
    "CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON tickets (created_at)",
    # No foreign key: notes may also belong to read-only archived tickets that are not stored here.
    """
    CREATE TABLE IF NOT EXISTS ticket_notes (
        note_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_id TEXT NOT NULL,
        at TEXT NOT NULL,
        author TEXT NOT NULL,
        text TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ticket_notes_ticket ON ticket_notes (ticket_id)",
    """
    CREATE TABLE IF NOT EXISTS idempotency_records (
        operation TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        request_hash TEXT NOT NULL,
        response_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (operation, idempotency_key)
    )
    """,
    # The counter only moves forward, so ids are never reused, even if rows were ever removed.
    """
    CREATE TABLE IF NOT EXISTS id_counter (
        name TEXT PRIMARY KEY,
        next_value INTEGER NOT NULL
    )
    """,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class TicketDatabase:
    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path)

    # ------------------------------------------------------------ connection helpers
    def _connect(self) -> sqlite3.Connection:
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.database_path, timeout=10, isolation_level=None)
            connection.row_factory = sqlite3.Row
            return connection
        except (OSError, sqlite3.Error) as exc:
            raise TicketDatabaseError("Could not open the helpdesk ticket database.") from exc

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    yield connection
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
                connection.execute("COMMIT")
        except sqlite3.Error as exc:
            raise TicketDatabaseError("The helpdesk ticket database operation failed.") from exc

    @contextmanager
    def _reader(self) -> Iterator[sqlite3.Connection]:
        try:
            with closing(self._connect()) as connection:
                yield connection
        except sqlite3.Error as exc:
            raise TicketDatabaseError("The helpdesk ticket database could not be read.") from exc

    def initialize(self) -> None:
        """Create tables if missing. Never drops, truncates or resets existing data."""
        with self._transaction() as connection:
            for statement in _SCHEMA:
                connection.execute(statement)
            connection.execute(
                "INSERT OR IGNORE INTO id_counter (name, next_value) VALUES ('ticket', ?)",
                (FIRST_CREATED_TICKET_NUMBER,),
            )

    # ------------------------------------------------------------ row mapping
    @staticmethod
    def _ticket_from_row(row: sqlite3.Row, notes: list[dict[str, str]]) -> dict[str, Any]:
        ticket: dict[str, Any] = {
            "ticket_id": row["ticket_id"],
            "subject": row["subject"],
            "description": row["description"],
            "category": row["category"],
            "priority": row["priority"],
            "status": row["status"],
            "requester_email": row["requester_email"],
            "department": row["department"],
            "affected_users": row["affected_users"],
            "assigned_team": row["assigned_team"],
            "attempted_steps": json.loads(row["attempted_steps_json"]),
            "kb_refs": json.loads(row["kb_refs_json"]),
            "resolution": row["resolution"],
            "created_at": row["created_at"],
            "resolved_at": row["resolved_at"],
            "notes": notes,
        }
        if row["warnings_json"]:
            ticket["warnings"] = json.loads(row["warnings_json"])
        return ticket

    @staticmethod
    def _note_from_row(row: sqlite3.Row) -> dict[str, str]:
        return {"at": row["at"], "author": row["author"], "text": row["text"]}

    # ------------------------------------------------------------ idempotency
    def get_idempotent(self, operation: str, key: str) -> dict[str, Any] | None:
        with self._reader() as connection:
            row = connection.execute(
                "SELECT request_hash, response_json FROM idempotency_records "
                "WHERE operation = ? AND idempotency_key = ?",
                (operation, key),
            ).fetchone()
        if row is None:
            return None
        return {"request_hash": row["request_hash"], "response": json.loads(row["response_json"])}

    @staticmethod
    def _idempotent_in_txn(connection: sqlite3.Connection, operation: str, key: str, request_hash: str):
        row = connection.execute(
            "SELECT request_hash, response_json FROM idempotency_records "
            "WHERE operation = ? AND idempotency_key = ?",
            (operation, key),
        ).fetchone()
        if row is None:
            return None
        if row["request_hash"] != request_hash:
            raise IdempotencyConflict("This Idempotency-Key was already used with a different request body.")
        return json.loads(row["response_json"])

    @staticmethod
    def _save_idempotent(connection, operation, key, request_hash, response) -> None:
        connection.execute(
            "INSERT INTO idempotency_records (operation, idempotency_key, request_hash, response_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (operation, key, request_hash, json.dumps(response), _utc_now()),
        )

    # ------------------------------------------------------------ tickets
    def create_ticket(
        self,
        fields: dict[str, Any],
        *,
        created_at: str,
        idempotency_key: str | None,
        request_hash: str,
    ) -> tuple[dict[str, Any], bool]:
        """Atomically allocate an id, store the ticket and its idempotency record.

        Returns (ticket, replayed). A repeated key with the same body returns the original
        response without creating anything; a different body raises IdempotencyConflict.
        """
        try:
            with self._transaction() as connection:
                if idempotency_key:
                    seen = self._idempotent_in_txn(connection, "create_ticket", idempotency_key, request_hash)
                    if seen is not None:
                        return seen, True

                number = connection.execute(
                    "SELECT next_value FROM id_counter WHERE name = 'ticket'"
                ).fetchone()["next_value"]
                connection.execute("UPDATE id_counter SET next_value = ? WHERE name = 'ticket'", (number + 1,))

                ticket_id = f"IT-{number}"
                warnings = fields.get("warnings") or None
                connection.execute(
                    "INSERT INTO tickets (ticket_number, " + _TICKET_COLUMNS + ") "
                    "VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?)",
                    (
                        number, ticket_id, fields["subject"], fields["description"], fields["category"],
                        fields["priority"], fields["requester_email"], fields["department"],
                        fields["affected_users"], fields["assigned_team"],
                        json.dumps(fields["attempted_steps"]), json.dumps(fields["kb_refs"]),
                        created_at, json.dumps(warnings) if warnings else None,
                    ),
                )
                row = connection.execute(
                    "SELECT " + _TICKET_COLUMNS + " FROM tickets WHERE ticket_id = ?", (ticket_id,)
                ).fetchone()
                ticket = self._ticket_from_row(row, [])
                if idempotency_key:
                    self._save_idempotent(connection, "create_ticket", idempotency_key, request_hash, ticket)
                return ticket, False
        except (KeyError, TypeError, ValueError) as exc:
            raise TicketDatabaseError("The ticket could not be stored.") from exc

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        with self._reader() as connection:
            row = connection.execute(
                "SELECT " + _TICKET_COLUMNS + " FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
            if row is None:
                return None
            notes = self._notes_for(connection, ticket_id)
        return self._ticket_from_row(row, notes)

    def list_tickets(self) -> list[dict[str, Any]]:
        """All database-backed tickets, oldest first, each with its notes."""
        with self._reader() as connection:
            rows = connection.execute(
                "SELECT " + _TICKET_COLUMNS + " FROM tickets ORDER BY ticket_number"
            ).fetchall()
            notes = self._all_notes(connection)
        return [self._ticket_from_row(row, notes.get(row["ticket_id"], [])) for row in rows]

    def count_tickets(self) -> int:
        with self._reader() as connection:
            return connection.execute("SELECT count(*) AS n FROM tickets").fetchone()["n"]

    # ------------------------------------------------------------ notes
    @classmethod
    def _notes_for(cls, connection, ticket_id: str) -> list[dict[str, str]]:
        rows = connection.execute(
            "SELECT at, author, text FROM ticket_notes WHERE ticket_id = ? ORDER BY note_id", (ticket_id,)
        ).fetchall()
        return [cls._note_from_row(row) for row in rows]

    @classmethod
    def _all_notes(cls, connection) -> dict[str, list[dict[str, str]]]:
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in connection.execute("SELECT ticket_id, at, author, text FROM ticket_notes ORDER BY note_id"):
            grouped.setdefault(row["ticket_id"], []).append(cls._note_from_row(row))
        return grouped

    def notes_by_ticket(self) -> dict[str, list[dict[str, str]]]:
        with self._reader() as connection:
            return self._all_notes(connection)

    def notes_for(self, ticket_id: str) -> list[dict[str, str]]:
        with self._reader() as connection:
            return self._notes_for(connection, ticket_id)

    def list_notes(self) -> list[dict[str, str]]:
        with self._reader() as connection:
            rows = connection.execute(
                "SELECT ticket_id, at, author, text FROM ticket_notes ORDER BY note_id"
            ).fetchall()
        return [{"ticket_id": row["ticket_id"], **self._note_from_row(row)} for row in rows]

    def add_note(
        self,
        ticket_id: str,
        *,
        at: str,
        author: str,
        text: str,
        archive_note_count: int,
        idempotency_key: str | None,
        request_hash: str,
    ) -> tuple[dict[str, Any], bool]:
        """Store one note. archive_note_count is the number of notes already in the read-only archive."""
        with self._transaction() as connection:
            if idempotency_key:
                seen = self._idempotent_in_txn(connection, "add_note", idempotency_key, request_hash)
                if seen is not None:
                    return seen, True
            connection.execute(
                "INSERT INTO ticket_notes (ticket_id, at, author, text) VALUES (?, ?, ?, ?)",
                (ticket_id, at, author, text),
            )
            stored = connection.execute(
                "SELECT count(*) AS n FROM ticket_notes WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()["n"]
            response = {
                "ticket_id": ticket_id,
                "note": {"at": at, "author": author, "text": text},
                "notes_count": archive_note_count + stored,
            }
            if idempotency_key:
                self._save_idempotent(connection, "add_note", idempotency_key, request_hash, response)
            return response, False

    # ------------------------------------------------------------ test-only reset
    def clear_all(self) -> None:
        """Delete every record and restart ids. Callers must guard this; only for isolated test databases."""
        with self._transaction() as connection:
            for table in ("tickets", "ticket_notes", "idempotency_records"):
                connection.execute(f"DELETE FROM {table}")
            connection.execute(
                "UPDATE id_counter SET next_value = ? WHERE name = 'ticket'", (FIRST_CREATED_TICKET_NUMBER,)
            )
