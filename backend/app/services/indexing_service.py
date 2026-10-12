from app.database.qdrant_client import (
    insert_chunk,
    insert_chunks,
)


def index_chunk(
    client,
    embedder,
    collection_name: str,
    chunk: dict,
) -> None:
    """
    Generate an embedding for one chunk
    and store it in Qdrant.
    """
    embedding = embedder.embed_text(chunk["text"])

    insert_chunk(
        client=client,
        collection_name=collection_name,
        chunk=chunk,
        embedding=embedding,
    )


def index_chunks(
    client,
    embedder,
    collection_name: str,
    chunks: list[dict],
    batch_size: int = 32,
) -> int:
    """
    Generate embeddings in batches and perform
    one Qdrant upsert per batch.

    Returns:
        Number of indexed chunks.
    """
    if not chunks:
        return 0

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero"
        )

    indexed_count = 0

    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[
            start:start + batch_size
        ]

        texts = [
            chunk["text"]
            for chunk in batch_chunks
        ]

        embeddings = embedder.embed_texts(texts)

        if len(embeddings) != len(batch_chunks):
            raise ValueError(
                "Number of embeddings must match "
                "number of chunks."
            )

        insert_chunks(
            client=client,
            collection_name=collection_name,
            chunks=batch_chunks,
            embeddings=embeddings,
        )

        indexed_count += len(batch_chunks)

    return indexed_count
