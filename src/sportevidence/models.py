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


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    embedding: list[float]
    document_id: str
    source_path: str
    distance: float


@dataclass(frozen=True, slots=True)
class RetrievedQueries:
    queries: list[str]
    retrieved_chunks: list[list[RetrievedChunk]]
