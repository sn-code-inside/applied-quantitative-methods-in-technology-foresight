from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Document:
    """Small text document used in the educational pipeline.

    The demo uses plain text files with optional metadata in the first lines:
    Title, Date, Source, and License. This avoids restricted databases and keeps
    the example easy to inspect.
    """

    doc_id: str
    title: str
    date: str
    source: str
    license: str
    text: str


def _read_header(lines: list[str], key: str, fallback: str = "") -> str:
    prefix = f"{key}:"
    for line in lines[:6]:
        if line.startswith(prefix):
            return line.replace(prefix, "", 1).strip()
    return fallback


def load_documents(folder: Path) -> List[Document]:
    """Load all .txt documents from a folder.

    Parameters
    ----------
    folder:
        Folder containing synthetic/open teaching documents.

    Returns
    -------
    list[Document]
        Parsed documents with metadata and body text.
    """

    docs: List[Document] = []
    for fp in sorted(folder.glob("*.txt")):
        raw = fp.read_text(encoding="utf-8").strip()
        lines = raw.splitlines()
        title = _read_header(lines, "Title", fp.stem)
        date = _read_header(lines, "Date", "")
        source = _read_header(lines, "Source", "")
        license_value = _read_header(lines, "License", "")

        body_start = 0
        for i, line in enumerate(lines):
            if line.strip() == "":
                body_start = i + 1
                break
        body = "\n".join(lines[body_start:]).strip() if body_start else raw

        docs.append(
            Document(
                doc_id=fp.stem,
                title=title,
                date=date,
                source=source,
                license=license_value,
                text=body,
            )
        )

    if not docs:
        raise SystemExit(f"No .txt documents found in {folder}")
    return docs
