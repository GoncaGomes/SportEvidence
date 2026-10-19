from pathlib import Path

from chromadb import chromadb

from sportevidence.models import Document


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
