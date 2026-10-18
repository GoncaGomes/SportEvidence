from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    source_path: Path
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    text: str
    embedding: list[float] | None = None
