import os
from dataclasses import replace
from math import isfinite
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from sportevidence.models import Document

load_dotenv()


CLIENT = OpenAI(
    base_url=os.getenv("SKYNET_BASE_URL"),
    api_key=os.getenv("SKYNET_API_KEY"),
    timeout=float(os.getenv("EMBEDDING_TIMEOUT_S", "30")),
    max_retries=0,
)

PDF_PATH = Path("tests/fixtures/SPORT-001.pdf")


def embed_texts(
    texts: list[str],
) -> list[list[float]]:

    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Texts must not be empty or whitespace-only.")

    response = CLIENT.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL"), input=texts, encoding_format="float"
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


def embed_chunks(doc: Document) -> Document:
    texts = [chunk.text for chunk in doc.chunks]
    vectors = embed_texts(texts)

    new_chunks = []

    for chunk, vector in zip(doc.chunks, vectors):
        new_chunk = replace(chunk, embedding=vector)
        new_chunks.append(new_chunk)

    return replace(doc, chunks=tuple(new_chunks))
