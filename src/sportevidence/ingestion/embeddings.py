from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from math import isfinite
from pathlib import Path

from openai import OpenAI

from sportevidence.config import EmbeddingSettings
from sportevidence.ingestion.chunking import (
    chunk_document,
)
from sportevidence.models import Chunk, Document


def embed_texts(
    texts: list[str],
    *,
    client: OpenAI | None = None,
    settings: EmbeddingSettings | None = None,
) -> list[list[float]]:
    """Embed texts, creating a client only when a request is needed."""
    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Texts must not be empty or whitespace-only.")

    settings = settings if settings is not None else EmbeddingSettings.from_env()
    if client is None:
        with OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout_s,
            max_retries=0,
        ) as owned_client:
            return embed_texts(texts, client=owned_client, settings=settings)

    response = client.embeddings.create(
        model=settings.model, input=texts, encoding_format="float"
    )
    results = sorted(response.data, key=lambda item: item.index)

    received_indices = [item.index for item in results]
    expected_indices = list(range(len(texts)))

    if received_indices != expected_indices:
        raise RuntimeError("Missing, duplicated or unexpected embedding indices.")

    dimension = len(results[0].embedding)

    for item in results:
        vector = item.embedding
        if len(vector) != dimension:
            raise RuntimeError("Embedding dimensions do not match.")
        if not all(isfinite(value) for value in vector):
            raise RuntimeError("An embedding contains non-finite values.")
        if not any(vector):
            raise RuntimeError("The server returned a zero vector.")

    return [item.embedding for item in results]


def embed_document(json_path: Path) -> Document:
    chunks, title = chunk_document(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        temp = json.load(f)
    document_id = temp["name"].replace("_prepared", "")
    source_path_hash = calculate_sha256(json_path)
    embeded_chunks = embed_texts(chunks)
    chunks = [
        Chunk(text=chunk_text, embedding=embeded_txt)
        for chunk_text, embeded_txt in zip(chunks, embeded_chunks)
    ]
    doc = Document(
        title=title,
        document_id=document_id,
        source_path=json_path,
        source_path_hash=source_path_hash,
        chunks=chunks,
    )
    return doc


def save_embeded_document_json(
    doc: Document,
    output_directory: Path = Path("tests/fixtures/embeded_documents"),
) -> None:
    data = asdict(doc)
    data["source_path"] = str(data["source_path"])
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{doc.document_id}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def calculate_sha256(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with Path(pdf_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
