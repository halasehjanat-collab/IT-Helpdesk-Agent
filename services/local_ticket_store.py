"""SQLite persistence for local references to externally created tickets."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TicketStoreError(RuntimeError):
    """Raised when local ticket metadata cannot be stored or read."""


class TicketMetadataConflict(TicketStoreError):
    """Raised when an idempotency key is associated with different data."""


def configured_database_path() -> Path:
    """Resolve the configured SQLite file path."""
    project_root = Path(__file__).resolve().parent.parent
    configured = os.getenv("HELPDESK_METADATA_DB_PATH")
    path = Path(configured).expanduser() if configured else (
        project_root / "services" / "data" / "helpdesk_metadata.sqlite3"
    )
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def payload_digest(payload: dict[str, Any]) -> str:
    """Hash a canonical serialization without retaining ticket content."""
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SQLiteTicketMetadataStore:
    """Persist only local references and idempotency metadata, not ticket bodies."""

    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.database_path,
                timeout=10,
            )
            connection.row_factory = sqlite3.Row
            return connection
        except (OSError, sqlite3.Error) as exc:
            raise TicketStoreError(
                "Could not open the local ticket metadata database."
            ) from exc

    def initialize(self) -> None:
        """Create the metadata table if it does not already exist."""
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ticket_references (
                        idempotency_key TEXT PRIMARY KEY,
                        external_ticket_id TEXT NOT NULL UNIQUE,
                        status TEXT NOT NULL
                            CHECK (status IN ('created')),
                        created_at TEXT NOT NULL,
                        payload_sha256 TEXT NOT NULL
                            CHECK (length(payload_sha256) = 64)
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                        idx_ticket_references_created_at
                    ON ticket_references (created_at)
                    """
                )
        except sqlite3.Error as exc:
            raise TicketStoreError(
                "Could not initialize the local ticket metadata database."
            ) from exc

    def record_created_ticket(
        self,
        *,
        idempotency_key: str,
        external_ticket_id: str,
        payload_sha256: str,
    ) -> dict[str, str]:
        """Record a successful external creation once per action key."""
        created_at = datetime.now(timezone.utc).isoformat()

        try:
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    """
                    SELECT idempotency_key, external_ticket_id, status,
                           created_at, payload_sha256
                    FROM ticket_references
                    WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                ).fetchone()

                if existing is not None:
                    row = dict(existing)
                    if (
                        row["external_ticket_id"] != external_ticket_id
                        or row["payload_sha256"] != payload_sha256
                    ):
                        raise TicketMetadataConflict(
                            "The action idempotency key is already associated "
                            "with different ticket metadata."
                        )
                    return row

                connection.execute(
                    """
                    INSERT INTO ticket_references (
                        idempotency_key, external_ticket_id, status,
                        created_at, payload_sha256
                    )
                    VALUES (?, ?, 'created', ?, ?)
                    """,
                    (
                        idempotency_key,
                        external_ticket_id,
                        created_at,
                        payload_sha256,
                    ),
                )
                row = connection.execute(
                    """
                    SELECT idempotency_key, external_ticket_id, status,
                           created_at, payload_sha256
                    FROM ticket_references
                    WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                ).fetchone()
                if row is None:
                    raise TicketStoreError(
                        "The ticket reference was not returned after insert."
                    )
                return dict(row)
        except TicketMetadataConflict:
            raise
        except sqlite3.Error as exc:
            raise TicketStoreError(
                "Could not persist the created ticket reference."
            ) from exc

    def get_ticket_reference(
        self,
        idempotency_key: str,
    ) -> dict[str, str] | None:
        try:
            with closing(self._connect()) as connection, connection:
                row = connection.execute(
                    """
                    SELECT idempotency_key, external_ticket_id, status,
                           created_at, payload_sha256
                    FROM ticket_references
                    WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                ).fetchone()
                return dict(row) if row is not None else None
        except sqlite3.Error as exc:
            raise TicketStoreError(
                "Could not read the local ticket metadata database."
            ) from exc

    def list_ticket_references(self) -> list[dict[str, str]]:
        try:
            with closing(self._connect()) as connection, connection:
                rows = connection.execute(
                    """
                    SELECT idempotency_key, external_ticket_id, status,
                           created_at, payload_sha256
                    FROM ticket_references
                    ORDER BY created_at, idempotency_key
                    """
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            raise TicketStoreError(
                "Could not read the local ticket metadata database."
            ) from exc

    def export_json(self, output_path: Path | str) -> Path:
        """Export metadata as JSON; the database remains the local source."""
        target = Path(output_path).expanduser()
        if not target.is_absolute():
            target = Path.cwd() / target
        target = target.resolve()
        export = {
            "format": "helpdesk-ticket-reference-export-v1",
            "records": self.list_ticket_references(),
        }
        temporary = target.with_name(
            f".{target.name}.{uuid.uuid4().hex}.tmp"
        )

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(export, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(target)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise TicketStoreError(
                "Could not write the ticket metadata JSON export."
            ) from exc

        return target
