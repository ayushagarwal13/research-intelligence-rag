import pytest

from app.retrieval.rrf import reciprocal_rank_fusion


def test_rrf_combines_results():
    dense_results = [
        {"chunk_id": "A", "text": "Chunk A"},
        {"chunk_id": "B", "text": "Chunk B"},
        {"chunk_id": "C", "text": "Chunk C"},
    ]

    keyword_results = [
        {"chunk_id": "C", "text": "Chunk C"},
        {"chunk_id": "A", "text": "Chunk A"},
        {"chunk_id": "D", "text": "Chunk D"},
    ]

    results = reciprocal_rank_fusion(
        retrieval_results=[
            dense_results,
            keyword_results,
        ]
    )

    assert len(results) == 4
    assert results[0]["chunk_id"] == "A"
    assert "rrf_score" in results[0]


def test_rrf_preserves_chunk_metadata():
    results = reciprocal_rank_fusion(
        retrieval_results=[
            [
                {
                    "chunk_id": "A",
                    "document_id": "doc-001",
                    "text": "Example text",
                }
            ]
        ]
    )

    assert results[0]["document_id"] == "doc-001"
    assert results[0]["text"] == "Example text"


def test_empty_results():
    results = reciprocal_rank_fusion([])

    assert results == []


def test_invalid_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            retrieval_results=[],
            k=0,
        )
