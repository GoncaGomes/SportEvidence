from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    source_path: Path
    source_hash: str
    chunks: list[Chunk]
    title: str | None = None


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    embedding: list[float] | None = None
