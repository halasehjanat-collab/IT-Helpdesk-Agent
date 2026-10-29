"""Load, parse, and chunk Markdown and PDF knowledge-base documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
import yaml


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


@dataclass(frozen=True)
class KnowledgeBaseDocument:
    metadata: dict[str, Any]
    body: str
    source_filename: str


@dataclass(frozen=True)
class KnowledgeBaseChunk:
    text: str
    metadata: dict[str, Any]
    source_filename: str
    chunk_index: int


def get_knowledge_base_dir() -> Path:
    """Return the project's knowledge-base directory."""
    return (
        Path(__file__).resolve().parent.parent
        / "dataset"
        / "knowledge_base"
    )


def _parse_yaml_metadata(
    text: str,
    filename: str,
) -> dict[str, Any]:
    """Parse and validate the presence of a YAML metadata mapping."""

    try:
        metadata = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"{filename}: invalid YAML metadata: {exc}"
        ) from exc

    if not isinstance(metadata, dict) or not metadata:
        raise ValueError(
            f"{filename}: metadata must be a non-empty mapping"
        )

    return dict(metadata)


def _parse_markdown(path: Path) -> KnowledgeBaseDocument:
    """Read Markdown and parse its YAML front matter."""

    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines(keepends=True)

    if not lines or lines[0].strip() != "---":
        raise ValueError(
            f"{path.name}: missing YAML front matter opening delimiter"
        )

    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        ),
        None,
    )

    if closing_index is None:
        raise ValueError(
            f"{path.name}: missing YAML front matter closing delimiter"
        )

    metadata_text = "".join(lines[1:closing_index])

    metadata = _parse_yaml_metadata(
        metadata_text,
        path.name,
    )

    body = "".join(
        lines[closing_index + 1:]
    ).strip()

    if not body:
        raise ValueError(
            f"{path.name}: document body is empty"
        )

    metadata.setdefault("source_filename", path.name)
    metadata.setdefault("source_type", "markdown")

    return KnowledgeBaseDocument(
        metadata=metadata,
        body=body,
        source_filename=path.name,
    )


def _extract_pdf_metadata(
    first_page_text: str,
    filename: str,
) -> dict[str, Any]:
    """Extract YAML metadata from the PDF DOCUMENT CONTROL section."""

    control_match = re.search(
        r"DOCUMENT\s+CONTROL\s*[\r\n]+(.*)",
        first_page_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not control_match:
        raise ValueError(
            f"{filename}: missing DOCUMENT CONTROL section"
        )

    control_text = control_match.group(1)

    delimiter_matches = list(
        re.finditer(
            r"(?m)^\s*---\s*$",
            control_text,
        )
    )

    if len(delimiter_matches) < 2:
        raise ValueError(
            f"{filename}: missing YAML metadata delimiters "
            "in DOCUMENT CONTROL"
        )

    metadata_text = control_text[
        delimiter_matches[0].end():
        delimiter_matches[1].start()
    ].strip()

    metadata = _parse_yaml_metadata(
        metadata_text,
        filename,
    )

    metadata["source_filename"] = filename
    metadata["source_type"] = "pdf"

    return metadata


def _parse_pdf(path: Path) -> list[KnowledgeBaseDocument]:
    """Extract text from each PDF page and preserve document metadata."""

    documents: list[KnowledgeBaseDocument] = []

    try:
        with fitz.open(path) as pdf:
            if len(pdf) == 0:
                raise ValueError(
                    f"{path.name}: PDF contains no pages"
                )

            first_page_text = pdf[0].get_text("text") or ""

            metadata = _extract_pdf_metadata(
                first_page_text,
                path.name,
            )

            for page_number, page in enumerate(pdf, start=1):
                text = (
                    page.get_text("text") or ""
                ).strip()

                if not text:
                    continue

                page_metadata = dict(metadata)
                page_metadata["page"] = page_number

                documents.append(
                    KnowledgeBaseDocument(
                        metadata=page_metadata,
                        body=text,
                        source_filename=path.name,
                    )
                )

    except fitz.FileDataError as exc:
        raise ValueError(
            f"{path.name}: could not read PDF: {exc}"
        ) from exc

    if not documents:
        raise ValueError(
            f"{path.name}: no extractable text found. "
            "The PDF may require OCR."
        )

    return documents


def load_markdown_documents(
    knowledge_base_dir: Path | None = None,
) -> list[KnowledgeBaseDocument]:
    """Load Markdown files while preserving their metadata."""

    if knowledge_base_dir is None:
        knowledge_base_dir = get_knowledge_base_dir()

    if not knowledge_base_dir.is_dir():
        raise FileNotFoundError(
            f"Knowledge-base directory not found: "
            f"{knowledge_base_dir}"
        )

    return [
        _parse_markdown(path)
        for path in sorted(
            knowledge_base_dir.glob("*.md")
        )
    ]


def load_knowledge_base(
    knowledge_base_dir: Path | None = None,
) -> list[KnowledgeBaseDocument]:
    """Load supported Markdown and PDF documents."""

    if knowledge_base_dir is None:
        knowledge_base_dir = get_knowledge_base_dir()

    if not knowledge_base_dir.is_dir():
        raise FileNotFoundError(
            f"Knowledge-base directory not found: "
            f"{knowledge_base_dir}"
        )

    documents: list[KnowledgeBaseDocument] = []

    for path in sorted(knowledge_base_dir.iterdir()):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix == ".md":
            documents.append(
                _parse_markdown(path)
            )

        elif suffix == ".pdf":
            documents.extend(
                _parse_pdf(path)
            )

    if not documents:
        raise RuntimeError(
            f"No readable Markdown or PDF documents found in "
            f"{knowledge_base_dir}"
        )

    return documents


def chunk_documents(
    documents: list[KnowledgeBaseDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[KnowledgeBaseChunk]:
    """Split documents into overlapping chunks and preserve metadata."""

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller "
            "than chunk_size"
        )

    chunks: list[KnowledgeBaseChunk] = []

    step = chunk_size - chunk_overlap

    for document in documents:
        text = document.body.strip()

        if not text:
            continue

        chunk_index = 0

        for start in range(0, len(text), step):
            chunk_text = text[
                start:start + chunk_size
            ].strip()

            if not chunk_text:
                continue

            metadata = dict(document.metadata)
            metadata["source_filename"] = (
                document.source_filename
            )
            metadata["chunk_index"] = chunk_index

            chunks.append(
                KnowledgeBaseChunk(
                    text=chunk_text,
                    metadata=metadata,
                    source_filename=document.source_filename,
                    chunk_index=chunk_index,
                )
            )

            chunk_index += 1

            if start + chunk_size >= len(text):
                break

    return chunks


def _self_check() -> None:
    """Run a basic ingestion and chunking check."""

    documents = load_knowledge_base()
    chunks = chunk_documents(documents)

    markdown_count = sum(
        document.metadata.get("source_type") == "markdown"
        for document in documents
    )

    pdf_pages = sum(
        document.metadata.get("source_type") == "pdf"
        for document in documents
    )

    current_count = sum(
        document.metadata.get("status") == "current"
        for document in documents
    )

    print(
        f"Loaded Markdown documents: {markdown_count}"
    )
    print(
        f"Loaded PDF pages with extractable text: {pdf_pages}"
    )
    print(
        f"Total document records: {len(documents)}"
    )
    print(
        f"Current document records: {current_count}"
    )
    print(
        f"Total chunks created: {len(chunks)}"
    )

    missing_metadata = [
        document.source_filename
        for document in documents
        if not document.metadata.get("id")
        or not document.metadata.get("status")
    ]

    if missing_metadata:
        raise ValueError(
            "Documents are missing required metadata: "
            + ", ".join(sorted(set(missing_metadata)))
        )


if __name__ == "__main__":
    _self_check()