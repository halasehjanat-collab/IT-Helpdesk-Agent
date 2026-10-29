"""Export the full ticket history held by the simulated helpdesk store to a JSON file.

Usage (from the kit folder):
    python -m services.export_ticket_history <output.json> [--include-archive]

This is a FULL ticket-history export (format helpdesk-ticket-history-export-v1): complete ticket
records and notes read from the helpdesk ticket database (HELPDESK_TICKET_DB_PATH). It is different
from the metadata-only export (helpdesk-ticket-reference-export-v1) made by
services.export_ticket_metadata. The historical seed archive is only read, never rewritten.
Every record is validated against $defs/ticket_record in ticket.schema.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator

from services.helpdesk_ticket_db import TicketDatabase, TicketDatabaseError, configured_ticket_db_path

KIT = Path(__file__).resolve().parent.parent
EXPORT_FORMAT = "helpdesk-ticket-history-export-v1"


def _dataset_dir() -> Path:
    configured = os.getenv("HELPDESK_DATASET_DIR")
    return Path(configured) if configured else KIT / "dataset"


def _data_dir() -> Path:
    configured = os.getenv("HELPDESK_DATA_DIR")
    return Path(configured) if configured else KIT / "services" / "data"


def build_export(database: TicketDatabase, *, include_archive: bool = False) -> dict:
    dataset = _dataset_dir()
    schema = json.loads((dataset / "schemas" / "ticket.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(
        {"$schema": schema.get("$schema", "https://json-schema.org/draft/2020-12/schema"),
         "$ref": "#/$defs/ticket_record", "$defs": schema["$defs"]}
    )

    tickets: list[dict] = []
    if include_archive:
        archive = json.loads((dataset / "tickets" / "historical_tickets.json").read_text(encoding="utf-8"))["tickets"]
        extra_notes = database.notes_by_ticket()
        for archived in archive:
            ticket = copy.deepcopy(archived)
            ticket["notes"] = list(ticket.get("notes", [])) + extra_notes.get(ticket["ticket_id"], [])
            tickets.append(ticket)
    tickets.extend(database.list_tickets())

    for ticket in tickets:
        problems = sorted(error.message for error in validator.iter_errors(ticket))
        if problems:
            raise ValueError(f"{ticket.get('ticket_id')} does not match ticket.schema.json: {problems[0]}")

    return {
        "format": EXPORT_FORMAT,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "includes_archive": include_archive,
        "ticket_count": len(tickets),
        "tickets": tickets,
    }


def write_export(database: TicketDatabase, output: Path, *, include_archive: bool = False) -> dict:
    export = build_export(database, include_archive=include_archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=output.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(export, handle, indent=2)
        os.replace(temp_name, output)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return export


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--include-archive", action="store_true",
                        help="also include the read-only historical seed tickets")
    args = parser.parse_args(argv)
    database = TicketDatabase(configured_ticket_db_path(KIT, _data_dir()))
    try:
        database.initialize()
        export = write_export(database, args.output, include_archive=args.include_archive)
    except (TicketDatabaseError, ValueError, OSError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print(f"Exported {export['ticket_count']} tickets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
