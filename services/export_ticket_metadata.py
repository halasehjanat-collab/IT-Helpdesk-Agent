"""Export locally stored ticket references as JSON."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from services.local_ticket_store import (
    SQLiteTicketMetadataStore,
    TicketStoreError,
    configured_database_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export local helpdesk ticket-reference metadata as JSON."
    )
    parser.add_argument(
        "output",
        type=Path,
        help="destination JSON file (for example, exports/tickets.json)",
    )
    args = parser.parse_args()

    store = SQLiteTicketMetadataStore(configured_database_path())
    try:
        store.initialize()
        output_path = store.export_json(args.output)
    except TicketStoreError as exc:
        logging.error("%s", exc)
        return 1

    print(f"Exported ticket-reference metadata to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
