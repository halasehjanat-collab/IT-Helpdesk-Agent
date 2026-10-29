import json

import pytest

from services.local_ticket_store import (
    SQLiteTicketMetadataStore,
    TicketMetadataConflict,
    TicketStoreError,
    payload_digest,
)


def test_initialize_and_persist_ticket_reference(tmp_path):
    database_path = tmp_path / "nested" / "metadata.sqlite3"
    store = SQLiteTicketMetadataStore(database_path)

    store.initialize()
    recorded = store.record_created_ticket(
        idempotency_key="action-1",
        external_ticket_id="IT-2001",
        payload_sha256=payload_digest({"subject": "Email unavailable"}),
    )
    reopened = SQLiteTicketMetadataStore(database_path)

    assert recorded["idempotency_key"] == "action-1"
    assert recorded["external_ticket_id"] == "IT-2001"
    assert recorded["status"] == "created"
    assert reopened.get_ticket_reference("action-1") == recorded


def test_recording_same_action_is_idempotent(tmp_path):
    store = SQLiteTicketMetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()
    digest = payload_digest({"subject": "Email unavailable"})

    first = store.record_created_ticket(
        idempotency_key="action-1",
        external_ticket_id="IT-2001",
        payload_sha256=digest,
    )
    second = store.record_created_ticket(
        idempotency_key="action-1",
        external_ticket_id="IT-2001",
        payload_sha256=digest,
    )

    assert second == first
    assert store.list_ticket_references() == [first]


def test_reusing_action_key_for_different_ticket_is_rejected(tmp_path):
    store = SQLiteTicketMetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()
    store.record_created_ticket(
        idempotency_key="action-1",
        external_ticket_id="IT-2001",
        payload_sha256=payload_digest({"subject": "Email unavailable"}),
    )

    with pytest.raises(TicketMetadataConflict):
        store.record_created_ticket(
            idempotency_key="action-1",
            external_ticket_id="IT-2002",
            payload_sha256=payload_digest({"subject": "Email unavailable"}),
        )


def test_json_export_contains_metadata_not_ticket_body(tmp_path):
    store = SQLiteTicketMetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()
    store.record_created_ticket(
        idempotency_key="action-1",
        external_ticket_id="IT-2001",
        payload_sha256=payload_digest(
            {"subject": "Sensitive ticket subject"}
        ),
    )

    output = store.export_json(tmp_path / "exports" / "tickets.json")
    exported = json.loads(output.read_text(encoding="utf-8"))

    assert exported["format"] == "helpdesk-ticket-reference-export-v1"
    assert exported["records"][0]["external_ticket_id"] == "IT-2001"
    assert "payload_sha256" in exported["records"][0]
    assert "Sensitive ticket subject" not in output.read_text(encoding="utf-8")
    assert "payload" not in exported["records"][0]


def test_database_errors_are_reported(tmp_path):
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("not a directory", encoding="utf-8")
    store = SQLiteTicketMetadataStore(parent_file / "metadata.sqlite3")

    with pytest.raises(TicketStoreError):
        store.initialize()
