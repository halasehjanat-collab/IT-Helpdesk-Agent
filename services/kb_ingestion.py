"""Load Markdown knowledge-base documents and their YAML front matter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class KnowledgeBaseDocument:
    metadata: dict[str, Any]
    body: str
    source_filename: str


def load_markdown_documents(
    knowledge_base_dir: Path | None = None,
) -> list[KnowledgeBaseDocument]:
    if knowledge_base_dir is None:
        knowledge_base_dir = (
            Path(__file__).resolve().parent.parent / "dataset" / "knowledge_base"
        )
    if not knowledge_base_dir.is_dir():
        raise FileNotFoundError(
            f"Knowledge-base directory not found: {knowledge_base_dir}"
        )

    documents = []
    for path in sorted(knowledge_base_dir.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        if not lines or lines[0].lstrip("\ufeff").rstrip("\r\n") != "---":
            raise ValueError(
                f"{path.name}: missing YAML front matter opening '---' delimiter"
            )

        closing_index = next(
            (
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.rstrip("\r\n") == "---"
            ),
            None,
        )
        if closing_index is None:
            raise ValueError(
                f"{path.name}: missing YAML front matter closing '---' delimiter"
            )

        front_matter = "".join(lines[1:closing_index])
        try:
            metadata = yaml.safe_load(front_matter)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"{path.name}: invalid YAML front matter: {exc}"
            ) from exc

        if not isinstance(metadata, dict) or not metadata:
            raise ValueError(
                f"{path.name}: YAML front matter must contain a metadata mapping"
            )

        body = "".join(lines[closing_index + 1 :]).lstrip("\r\n")
        documents.append(
            KnowledgeBaseDocument(
                metadata=metadata,
                body=body,
                source_filename=path.name,
            )
        )

    return documents


def _self_check() -> None:
    documents = load_markdown_documents()
    if not documents:
        raise RuntimeError("No Markdown knowledge-base documents were found.")

    for document in documents:
        if not document.metadata:
            raise ValueError(f"{document.source_filename}: metadata is empty")
        if not document.body.strip():
            raise ValueError(f"{document.source_filename}: document body is empty")
        if not document.source_filename:
            raise ValueError("A loaded document has no source filename")

    print(f"OK: loaded and checked {len(documents)} Markdown documents.")


if __name__ == "__main__":
    _self_check()
