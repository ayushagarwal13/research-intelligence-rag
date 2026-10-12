from app.database.qdrant_client import (
    create_collection,
    create_qdrant_client,
)
from app.services.indexing_service import index_chunks


class FakeBatchEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]


def test_index_chunks():
    client = create_qdrant_client()
    collection_name = "batch_documents"

    create_collection(client, collection_name)

    chunks = [
        {
            "chunk_id": "batch-001",
            "document_id": "document-001",
            "document_name": "paper.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "Machine learning uses data.",
        },
        {
            "chunk_id": "batch-002",
            "document_id": "document-001",
            "document_name": "paper.pdf",
            "page_number": 1,
            "chunk_index": 1,
            "text": "Neural networks learn patterns.",
        },
    ]

    embedder = FakeBatchEmbedder()

    indexed_count = index_chunks(
        client=client,
        embedder=embedder,
        collection_name=collection_name,
        chunks=chunks,
    )

    assert indexed_count == 2

    stored_count = client.count(
        collection_name=collection_name,
        exact=True,
    ).count

    assert stored_count == 2


def test_index_chunks_respects_batch_size():
    client = create_qdrant_client()
    collection_name = "batch_size_documents"

    create_collection(client, collection_name)

    chunks = [
        {
            "chunk_id": f"batch-size-{index}",
            "document_id": "document-001",
            "document_name": "paper.pdf",
            "page_number": 1,
            "chunk_index": index,
            "text": f"Research content {index}.",
        }
        for index in range(5)
    ]

    class TrackingEmbedder:
        def __init__(self):
            self.batch_sizes = []

        def embed_texts(
            self,
            texts: list[str],
        ) -> list[list[float]]:
            self.batch_sizes.append(len(texts))
            return [[0.1] * 768 for _ in texts]

    embedder = TrackingEmbedder()

    indexed_count = index_chunks(
        client=client,
        embedder=embedder,
        collection_name=collection_name,
        chunks=chunks,
        batch_size=2,
    )

    assert indexed_count == 5
    assert embedder.batch_sizes == [2, 2, 1]
