import pytest

from app.reranking.reranker import CrossEncoderReranker


@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker()


def test_reranker_returns_ranked_results(reranker):
    chunks = [
        {
            "chunk_id": "chunk-001",
            "text": (
                "RAG systems retrieve relevant "
                "documents before generating answers."
            ),
        },
        {
            "chunk_id": "chunk-002",
            "text": (
                "Convolutional neural networks "
                "are commonly used for image processing."
            ),
        },
    ]

    results = reranker.rerank(
        query="How does retrieval augmented generation work?",
        chunks=chunks,
        top_k=2,
    )

    assert len(results) == 2

    assert all(
        "rerank_score" in result
        for result in results
    )


def test_reranker_preserves_metadata(reranker):
    chunks = [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "page_number": 5,
            "text": "RAG retrieves external context.",
        }
    ]

    results = reranker.rerank(
        query="What does RAG retrieve?",
        chunks=chunks,
        top_k=1,
    )

    assert results[0]["chunk_id"] == "chunk-001"
    assert results[0]["document_id"] == "doc-001"
    assert results[0]["page_number"] == 5


def test_empty_chunks(reranker):
    results = reranker.rerank(
        query="What is RAG?",
        chunks=[],
    )

    assert results == []


def test_empty_query_raises_error(reranker):
    with pytest.raises(ValueError):
        reranker.rerank(
            query="",
            chunks=[
                {
                    "chunk_id": "chunk-001",
                    "text": "Some text.",
                }
            ],
        )


def test_invalid_top_k_raises_error(reranker):
    with pytest.raises(ValueError):
        reranker.rerank(
            query="What is RAG?",
            chunks=[
                {
                    "chunk_id": "chunk-001",
                    "text": "Some text.",
                }
            ],
            top_k=0,
        )
