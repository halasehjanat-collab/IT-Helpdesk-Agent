import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services import export_ticket_history
from services.helpdesk_ticket_db import TicketDatabase, TicketDatabaseError
from tests.conftest import VALID_PAYLOAD as _APP_PAYLOAD

# The API validates against the real directory, so use a directory employee.
VALID_PAYLOAD = {**_APP_PAYLOAD, "requester_email": "samir.nassar@nakhla.example", "department": "Executive Office"}

DEFAULT_DB = Path(__file__).resolve().parent.parent / "services" / "data" / "helpdesk_tickets.sqlite3"


def body(**overrides):
    return {**VALID_PAYLOAD, **overrides}


def create(client, key=None, **overrides):
    headers = {"Idempotency-Key": key} if key else {}
    return client.post("/api/tickets", json=body(**overrides), headers=headers)


def test_ticket_and_notes_survive_restart(ticket_api):
    first = ticket_api()
    created = create(first, key="k1")
    assert created.status_code == 201
    ticket_id = created.json()["ticket_id"]
    assert ticket_id == "IT-2001"
    note = first.post(f"/api/tickets/{ticket_id}/notes", json={"note": "Called the user."})
    assert note.status_code == 201 and note.json()["notes_count"] == 1

    second = ticket_api()  # restart
    fetched = second.get(f"/api/tickets/{ticket_id}")
    assert fetched.status_code == 200
    stored = fetched.json()
    assert stored["subject"] == VALID_PAYLOAD["subject"]
    assert stored["status"] == "open" and stored["resolution"] is None
    assert [n["text"] for n in stored["notes"]] == ["Called the user."]


def test_ticket_ids_are_unique_across_restarts(ticket_api):
    ids = []
    for _ in range(3):
        client = ticket_api()
        ids.append(create(client).json()["ticket_id"])
    assert ids == ["IT-2001", "IT-2002", "IT-2003"]


def test_idempotency_survives_restart(ticket_api):
    first = ticket_api()
    original = create(first, key="same").json()
    second = ticket_api()
    replay = create(second, key="same")
    assert replay.status_code == 201
    assert replay.headers["Idempotent-Replayed"] == "true"
    assert replay.json()["ticket_id"] == original["ticket_id"]
    assert len(second.get("/admin/tickets").json()["tickets"]) == 1
    conflict = create(second, key="same", subject="A different subject")
    assert conflict.status_code == 409
    assert len(second.get("/admin/tickets").json()["tickets"]) == 1


def test_note_idempotency_survives_restart(ticket_api):
    first = ticket_api()
    tid = create(first).json()["ticket_id"]
    headers = {"Idempotency-Key": "n1"}
    first.post(f"/api/tickets/{tid}/notes", json={"note": "one"}, headers=headers)
    second = ticket_api()
    replay = second.post(f"/api/tickets/{tid}/notes", json={"note": "one"}, headers=headers)
    assert replay.headers["Idempotent-Replayed"] == "true"
    assert len(second.get(f"/api/tickets/{tid}").json()["notes"]) == 1
    changed = second.post(f"/api/tickets/{tid}/notes", json={"note": "two"}, headers=headers)
    assert changed.status_code == 409


def test_search_and_get_span_archive_and_database(ticket_api):
    client = ticket_api()
    new_id = create(client).json()["ticket_id"]
    client = ticket_api()
    everything = client.get("/api/tickets", params={"limit": 50, "requester_email": VALID_PAYLOAD["requester_email"]})
    ids = {t["ticket_id"] for t in everything.json()["tickets"]}
    assert new_id in ids
    archived = client.get("/admin/tickets", params={"all": 1}).json()["tickets"]
    assert any(t["ticket_id"] == "IT-1001" for t in archived)
    assert any(t["ticket_id"] == new_id for t in archived)
    assert client.get("/api/tickets/IT-1001").status_code == 200
    assert client.get("/api/tickets/it-2001").status_code == 200


def test_notes_on_archive_tickets_persist_without_changing_the_archive(ticket_api):
    module = ticket_api.module
    seed = module.DATASET_DIR / "tickets" / "historical_tickets.json"
    before = seed.read_bytes()
    first = ticket_api()
    original_count = len(first.get("/api/tickets/IT-1001").json()["notes"])
    resp = first.post("/api/tickets/IT-1001/notes", json={"note": "Follow-up"})
    assert resp.json()["notes_count"] == original_count + 1
    second = ticket_api()
    notes = second.get("/api/tickets/IT-1001").json()["notes"]
    assert len(notes) == original_count + 1 and notes[-1]["text"] == "Follow-up"
    assert seed.read_bytes() == before


def test_reset_preserves_persisted_tickets_by_default(ticket_api):
    client = ticket_api()
    create(client)
    result = client.post("/admin/reset").json()
    assert result["persistent_tickets_preserved"] is True
    assert result["created_tickets"] == 1
    assert client.get("/api/tickets/IT-2001").status_code == 200
    assert create(client).json()["ticket_id"] == "IT-2002"


def test_destructive_reset_requires_explicit_flag_and_path(ticket_api, monkeypatch):
    client = ticket_api()
    create(client)
    monkeypatch.setenv("HELPDESK_ALLOW_DESTRUCTIVE_RESET", "1")
    monkeypatch.delenv("HELPDESK_TICKET_DB_PATH")
    assert client.post("/admin/reset").json()["persistent_tickets_preserved"] is True
    assert client.get("/api/tickets/IT-2001").status_code == 200

    monkeypatch.setenv("HELPDESK_TICKET_DB_PATH", str(ticket_api.db_path))
    result = client.post("/admin/reset").json()
    assert result["persistent_tickets_preserved"] is False and result["created_tickets"] == 0
    assert client.get("/api/tickets/IT-2001").status_code == 404


def test_startup_never_deletes_or_resets(ticket_api):
    create(ticket_api())
    for _ in range(3):
        client = ticket_api()
    assert client.get("/admin/tickets").json()["count"] == 1
    assert create(client).json()["ticket_id"] == "IT-2002"


def test_default_database_is_not_touched_by_tests(ticket_api):
    existed = DEFAULT_DB.exists()
    create(ticket_api())
    assert DEFAULT_DB.exists() == existed


def test_database_failure_returns_503(ticket_api, monkeypatch):
    client = ticket_api()
    create(client)

    def broken(self):
        raise TicketDatabaseError("boom")

    monkeypatch.setattr(TicketDatabase, "_connect", broken)
    response = create(client)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "storage_unavailable"


def test_failed_write_rolls_back_and_does_not_burn_an_id(ticket_api, monkeypatch):
    client = ticket_api()

    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(TicketDatabase, "_save_idempotent", staticmethod(broken))
        assert create(client, key="k").status_code == 503
    assert client.get("/admin/tickets").json()["count"] == 0
    assert create(client, key="k").json()["ticket_id"] == "IT-2001"


def test_invalid_ticket_is_rejected_and_not_stored(ticket_api):
    client = ticket_api()
    response = create(client, category="not_a_category")
    assert response.status_code == 422
    assert client.get("/admin/tickets").json()["count"] == 0


def test_ticket_db_may_not_be_the_metadata_db(ticket_api, monkeypatch):
    from services.local_ticket_store import configured_database_path

    monkeypatch.setenv("HELPDESK_TICKET_DB_PATH", str(configured_database_path()))
    ticket_api.module.TICKET_DB = None
    with pytest.raises(TicketDatabaseError):
        ticket_api.module.ticket_db()


def test_api_still_requires_api_key(ticket_api):
    client = ticket_api()
    assert TestClient(ticket_api.module.app).post("/api/tickets", json=body()).status_code == 401
    assert client.get("/admin/tickets").status_code == 200


def test_history_export_is_full_valid_and_distinct(ticket_api, tmp_path):
    client = ticket_api()
    tid = create(client).json()["ticket_id"]
    client.post(f"/api/tickets/{tid}/notes", json={"note": "n"})
    output = tmp_path / "history.json"
    database = TicketDatabase(ticket_api.db_path)
    export = export_ticket_history.write_export(database, output)
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["format"] == "helpdesk-ticket-history-export-v1" == export["format"]
    assert [t["ticket_id"] for t in data["tickets"]] == [tid]
    assert data["tickets"][0]["description"] == VALID_PAYLOAD["description"]
    assert data["tickets"][0]["notes"][0]["text"] == "n"

    with_archive = export_ticket_history.build_export(database, include_archive=True)
    assert with_archive["ticket_count"] > 1 and with_archive["includes_archive"] is True
