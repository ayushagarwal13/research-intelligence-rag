from app.services.indexing_service import (
    index_chunk,
    index_chunks,
)
from app.database.qdrant_client import create_collection
from qdrant_client import QdrantClient
import pytest
from unittest.mock import Mock, call, patch


class FakeEmbedder:
    def embed_text(self, text: str) -> list[float]:
        return [0.1] * 768

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]


def test_index_chunk():
    # In-memory storage keeps this test independent of Qdrant Cloud.
    client = QdrantClient(":memory:")
    collection_name = "research_documents"

    try:
        create_collection(
            client,
            collection_name,
        )

        chunk = {
            "chunk_id": "chunk-service-001",
            "document_id": "document-001",
            "document_name": "research_paper.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "Neural networks learn patterns from data.",
        }

        embedder = FakeEmbedder()

        index_chunk(
            client=client,
            embedder=embedder,
            collection_name=collection_name,
            chunk=chunk,
        )

        assert client.count(
            collection_name=collection_name,
            exact=True,
        ).count == 1

    finally:
        client.close()


def test_index_chunks_returns_zero_for_empty_input():
    embedder = Mock()

    result = index_chunks(
        client=Mock(),
        embedder=embedder,
        collection_name="research_documents",
        chunks=[],
    )

    assert result == 0
    embedder.embed_texts.assert_not_called()


@pytest.mark.parametrize("batch_size", [0, -1])
def test_index_chunks_rejects_invalid_batch_size(batch_size):
    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero",
    ):
        index_chunks(
            client=Mock(),
            embedder=Mock(),
            collection_name="research_documents",
            chunks=[{"text": "Sample text"}],
            batch_size=batch_size,
        )


@patch("app.services.indexing_service.insert_chunks")
def test_index_chunks_processes_multiple_batches(
    mock_insert_chunks,
):
    chunks = [
        {
            "chunk_id": "chunk-1",
            "text": "First chunk",
        },
        {
            "chunk_id": "chunk-2",
            "text": "Second chunk",
        },
        {
            "chunk_id": "chunk-3",
            "text": "Third chunk",
        },
    ]

    embedder = FakeEmbedder()
    client = Mock()

    result = index_chunks(
        client=client,
        embedder=embedder,
        collection_name="research_documents",
        chunks=chunks,
        batch_size=2,
    )

    assert result == 3

    assert mock_insert_chunks.call_count == 2
    assert mock_insert_chunks.call_args_list == [
        call(
            client=client,
            collection_name="research_documents",
            chunks=chunks[:2],
            embeddings=[
                [0.1] * 768,
                [0.1] * 768,
            ],
        ),
        call(
            client=client,
            collection_name="research_documents",
            chunks=chunks[2:],
            embeddings=[
                [0.1] * 768,
            ],
        ),
    ]


@patch("app.services.indexing_service.insert_chunks")
def test_index_chunks_rejects_embedding_count_mismatch(
    mock_insert_chunks,
):
    chunks = [
        {
            "chunk_id": "chunk-1",
            "text": "First chunk",
        },
        {
            "chunk_id": "chunk-2",
            "text": "Second chunk",
        },
    ]

    embedder = Mock()
    embedder.embed_texts.return_value = [
        [0.1] * 768
    ]

    with pytest.raises(
        ValueError,
        match="Number of embeddings must match",
    ):
        index_chunks(
            client=Mock(),
            embedder=embedder,
            collection_name="research_documents",
            chunks=chunks,
        )

    mock_insert_chunks.assert_not_called()

