from app.services.reranking_service import rerank_results
import pytest


class FakeReranker:
    def __init__(self):
        self.called = False

    def rerank(self, query, chunks, top_k):
        self.called = True
        return chunks[:top_k]


def test_rerank_results():
    candidates = [
        {
            "chunk_id": "chunk-001",
            "text": "RAG retrieves context.",
        },
        {
            "chunk_id": "chunk-002",
            "text": "BERT processes language.",
        },
    ]

    reranker = FakeReranker()

    results = rerank_results(
        reranker=reranker,
        query="What is RAG?",
        candidates=candidates,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-001"
    assert reranker.called is True


@pytest.mark.parametrize("query", ["", " ", "   ", "\t"])
def test_rerank_results_rejects_empty_query(query):
    reranker = FakeReranker()

    with pytest.raises(
        ValueError,
        match="Query cannot be empty.",
    ):
        rerank_results(
            reranker=reranker,
            query=query,
            candidates=[],
            top_k=5,
        )

    assert reranker.called is False


@pytest.mark.parametrize("top_k", [0, -1, -5])
def test_rerank_results_rejects_invalid_top_k(top_k):
    reranker = FakeReranker()

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero.",
    ):
        rerank_results(
            reranker=reranker,
            query="What is RAG?",
            candidates=[],
            top_k=top_k,
        )

    assert reranker.called is False


