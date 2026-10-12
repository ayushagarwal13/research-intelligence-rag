from app.database.qdrant_client import insert_chunk


def index_chunk(
    client,
    embedder,
    collection_name: str,
    chunk: dict,
) -> None:
    """
    Generate an embedding for one chunk and store it in Qdrant.
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
    Generate embeddings in configurable batches
    and store chunks in Qdrant.

    Args:
        client: Qdrant client.
        embedder: Embedding model.
        collection_name: Target Qdrant collection.
        chunks: Chunks to index.
        batch_size: Number of chunks embedded per batch.

    Returns:
        Number of indexed chunks.
    """
    if not chunks:
        return 0

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    indexed_count = 0

    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start:start + batch_size]

        texts = [
            chunk["text"]
            for chunk in batch_chunks
        ]

        embeddings = embedder.embed_texts(texts)

        if len(embeddings) != len(batch_chunks):
            raise ValueError(
                "Number of embeddings must match number of chunks."
            )

        for chunk, embedding in zip(batch_chunks, embeddings):
            insert_chunk(
                client=client,
                collection_name=collection_name,
                chunk=chunk,
                embedding=embedding,
            )

            indexed_count += 1

    return indexed_count
