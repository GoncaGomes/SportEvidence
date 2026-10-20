from __future__ import annotations

from dataclasses import replace
from math import isfinite

from openai import OpenAI

from sportevidence.config import EmbeddingSettings
from sportevidence.models import Document


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


def embed_chunks(
    doc: Document,
    *,
    client: OpenAI | None = None,
    settings: EmbeddingSettings | None = None,
) -> Document:
    texts = [chunk.text for chunk in doc.chunks]
    vectors = embed_texts(texts, client=client, settings=settings)
    return replace(
        doc,
        chunks=tuple(
            replace(chunk, embedding=vector)
            for chunk, vector in zip(doc.chunks, vectors, strict=True)
        ),
    )
