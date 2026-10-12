from app.chunking.document_chunker import chunk_document
from app.services.indexing_service import index_chunks


def index_document(
    client,
    embedder,
    collection_name: str,
    document: dict,
    chunk_size: int = 500,
    overlap: int = 50,
) -> int:
    """
    Chunk a document and index all chunks using batch embeddings.
    """
    chunks = chunk_document(
        document=document,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return index_chunks(
        client=client,
        embedder=embedder,
        collection_name=collection_name,
        chunks=chunks,
    )
