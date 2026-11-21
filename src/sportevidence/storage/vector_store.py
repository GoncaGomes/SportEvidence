from pathlib import Path
from typing import cast

from chromadb import Collection

from sportevidence.models import Chunk, Document, SearchHit


def upsert_document(collection: Collection, document: Document) -> int:
    if not document.chunks:
        raise ValueError(f"Document {document.document_id!r} has no chunks.")

    ids = []
    texts = []
    embeddings = []
    metadatas = []

    document_metadata = {
        "document_id": document.document_id,
        "title": document.title,
        "source_path": str(document.source_path),
        "source_hash": document.source_hash,
    }

    for chunk in document.chunks:
        if chunk.embedding is None:
            raise ValueError(f"Chunk {chunk.chunk_id!r} has no embedding.")

        ids.append(chunk.chunk_id)
        texts.append(chunk.text)
        embeddings.append(chunk.embedding)
        metadatas.append(
            {
                **document_metadata,
                "chunk_index": chunk.chunk_index,
                "page_numbers": list(chunk.page_numbers) or None,
            }
        )

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(ids)


def get_document(collection: Collection, document_id: str) -> Document | None:
    result = collection.get(
        where={"document_id": document_id},
        include=["documents", "embeddings", "metadatas"],
    )
    if not result["ids"]:
        return None

    texts = result["documents"]
    embeddings = result["embeddings"]
    metadatas = result["metadatas"]

    if texts is None or embeddings is None or not metadatas:
        raise ValueError(f"Document {document_id!r} has incomplete records.")

    document_metadata = metadatas[0]
    if (
        document_metadata is None
        or "source_path" not in document_metadata
        or "source_hash" not in document_metadata
    ):
        raise ValueError(f"Document {document_id!r} has incomplete source metadata.")

    chunks = []
    for chunk_id, text, embedding, metadata in zip(
        result["ids"], texts, embeddings, metadatas, strict=True
    ):
        if metadata is None or "chunk_index" not in metadata:
            raise ValueError(f"Chunk {chunk_id!r} is missing its chunk_index.")

        chunk_index = metadata.get("chunk_index")
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
            raise TypeError(f"Chunk {chunk_id!r} must have an integer chunk_index.")

        raw_pages = metadata.get("page_numbers")
        pages = _read_page_numbers(raw_pages, chunk_id)

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                text=text,
                page_numbers=pages,
                embedding=[float(value) for value in embedding],
            )
        )

    chunks.sort(key=lambda chunk: chunk.chunk_index)

    title = document_metadata.get("title")
    if title is not None and not isinstance(title, str):
        raise ValueError(f"Document {document_id!r} has no title.")

    source_path = document_metadata.get("source_path")
    if source_path is None or not isinstance(source_path, str):
        raise ValueError(f"Document {document_id!r} has no source_path.")

    source_hash = document_metadata.get("source_hash")
    if source_hash is None or not isinstance(source_hash, str):
        raise ValueError(f"Document {document_id!r} has no source_hash.")

    return Document(
        document_id=document_id,
        source_path=Path(source_path),
        source_hash=source_hash,
        chunks=chunks,
        title=title,
    )


def search_chunks(
    collection: Collection,
    query_embedding: list[float],
    *,
    limit: int = 5,
    document_id: str | None = None,
) -> list[SearchHit]:
    if limit < 1:
        raise ValueError("Limit must be at least 1.")
    if not query_embedding:
        raise ValueError("Query embedding must not be empty.")

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        where={"document_id": document_id} if document_id is not None else None,
        include=["documents", "metadatas", "distances"],
    )

    ids = result["ids"][0]
    if not ids:
        return []

    texts = result["documents"]
    metadatas = result["metadatas"]
    distances = result["distances"]

    if texts is None or metadatas is None or distances is None:
        raise ValueError("Chroma returned incomplete search result.")

    hits = []
    for chunk_id, text, metadata, distance in zip(
        ids, texts[0], metadatas[0], distances[0], strict=True
    ):
        if metadata is None:
            raise ValueError(f"Chunk {chunk_id!r} has no metadata.")

        hit_document_id = metadata.get("document_id")
        source_path = metadata.get("source_path")
        if not isinstance(hit_document_id, str) or not isinstance(source_path, str):
            raise TypeError(f"Chunk {chunk_id!r} has invalid source metadata.")

        hits.append(
            SearchHit(
                chunk_id=chunk_id,
                document_id=hit_document_id,
                text=text,
                source_path=Path(source_path),
                page_numbers=_read_page_numbers(metadata.get("page_numbers"), chunk_id),
                distance=distance,
            )
        )
    return hits


def _read_page_numbers(raw_pages: object, chunk_id: str) -> tuple[int, ...]:
    if raw_pages is None:
        return ()

    if not isinstance(raw_pages, list) or not all(
        type(page) is int for page in raw_pages
    ):
        raise ValueError(f"Chunk {chunk_id!r} must have a list of integer pages.")

    return tuple(cast(list[int], raw_pages))
