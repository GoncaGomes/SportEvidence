"""Application settings loaded explicitly from the process environment."""

import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    model: str
    api_key: str
    base_url: str | None = None
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("EMBEDDING_MODEL must be set.")
        if not self.api_key.strip():
            raise ValueError("SKYNET_API_KEY must be set.")
        if not isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("EMBEDDING_TIMEOUT_S must be finite and positive.")

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        return cls(
            model=os.getenv("EMBEDDING_MODEL", ""),
            api_key=os.getenv("SKYNET_API_KEY", ""),
            base_url=os.getenv("SKYNET_BASE_URL") or None,
            timeout_s=float(os.getenv("EMBEDDING_TIMEOUT_S", "30")),
        )


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    db_path: Path = Path("data/chroma")
    collection_name: str = "default_collection"

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        return cls(
            db_path=Path(os.getenv("SPORTEVIDENCE_DB_PATH", "data/chroma")),
            collection_name=os.getenv("SPORTEVIDENCE_COLLECTION", "default_collection"),
        )
