from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Document:
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True, slots=True)
class Chunk:
    document_id: str
    chunk_id: str
    source_path: Path
    text: str
