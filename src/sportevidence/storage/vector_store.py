from pathlib import Path

import chromadb

from sportevidence.ingestion.embeddings import embed_texts
from sportevidence.models import Document, RetrievedChunk, RetrievedQueries


def upsert_document(doc: Document, db_path: Path, collection_name: str) -> None:
    db_path = db_path.resolve()
    if not doc.chunks:
        return

    with chromadb.PersistentClient(path=str(db_path)) as client:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )
        ids = []
        texts = []
        vectors = []
        metadata = []

        for chunk in doc.chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id!r} has no embedding")

            ids.append(chunk.chunk_id)
            texts.append(chunk.text)
            vectors.append(chunk.embedding)

            metadata.append(
                {
                    "document_id": doc.document_id,
                    "source_path": str(doc.source_path),
                }
            )

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metadata,
        )


def query_collection(
    query: list[str],
    db_path: Path,
    collection_name: str,
    n_results: int = 5,
) -> RetrievedQueries:
    """Return a group of retrieved chunks for each input query, in order."""
    if n_results <= 0:
        raise ValueError("n_results must be positive.")

    if not query:
        return RetrievedQueries(queries=[], retrieved_chunks=[])

    query_embeddings = embed_texts(query)

    client = chromadb.PersistentClient(path=str(db_path.resolve()))
    collection = client.get_collection(
        name=collection_name,
        embedding_function=None,
    )

    result = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        include=["documents", "embeddings", "metadatas", "distances"],
    )

    retrieved_chunks = []

    for query_index, chunk_ids in enumerate(result["ids"]):
        query_chunks = []

        for chunk_index, chunk_id in enumerate(chunk_ids):
            metadata = result["metadatas"][query_index][chunk_index]
            vector = result["embeddings"][query_index][chunk_index]

            query_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=result["documents"][query_index][chunk_index],
                    embedding=[float(value) for value in vector],
                    document_id=metadata["document_id"],
                    source_path=metadata["source_path"],
                    distance=float(result["distances"][query_index][chunk_index]),
                )
            )
        retrieved_chunks.append(query_chunks)

    return RetrievedQueries(
        queries=list(query),
        retrieved_chunks=retrieved_chunks,
    )
