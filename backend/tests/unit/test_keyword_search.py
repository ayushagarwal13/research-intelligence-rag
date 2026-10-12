import pytest

from app.retrieval.keyword_search import KeywordSearch


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "chunk-001",
            "document_id": "document-001",
            "text": (
                "BERT is a transformer-based "
                "language model for NLP."
            ),
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "document-001",
            "text": (
                "Retrieval augmented generation "
                "combines retrieval and generation."
            ),
        },
        {
            "chunk_id": "chunk-003",
            "document_id": "document-002",
            "text": (
                "Convolutional neural networks "
                "are used for image processing."
            ),
        },
    ]


def test_keyword_search_returns_results(
    sample_chunks,
):
    searcher = KeywordSearch(sample_chunks)

    results = searcher.search(
        query="BERT NLP",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-001"
    assert "keyword_score" in results[0]


def test_keyword_search_preserves_metadata(
    sample_chunks,
):
    searcher = KeywordSearch(sample_chunks)

    results = searcher.search(
        query="retrieval generation",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-002"
    assert results[0]["document_id"] == "document-001"


def test_empty_chunks_raise_error():
    with pytest.raises(ValueError):
        KeywordSearch([])


def test_empty_query_raises_error(
    sample_chunks,
):
    searcher = KeywordSearch(sample_chunks)

    with pytest.raises(ValueError):
        searcher.search("")


def test_invalid_top_k_raises_error(
    sample_chunks,
):
    searcher = KeywordSearch(sample_chunks)

    with pytest.raises(ValueError):
        searcher.search(
            query="BERT",
            top_k=0,
        )
