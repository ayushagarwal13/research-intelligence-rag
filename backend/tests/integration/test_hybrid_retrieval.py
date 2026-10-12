import pytest
from unittest.mock import patch
from app.database.qdrant_client import (
    create_collection,
    create_qdrant_client,
    insert_chunks,
)
from app.retrieval.hybrid_retrieval import hybrid_search
from app.retrieval.keyword_search import KeywordSearch


COLLECTION_NAME = "test_hybrid"


class FakeEmbedder:
    def embed_text(self, text: str) -> list[float]:
        return [0.1] * 768


def test_hybrid_search():

    chunks = [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "document_name": "rag.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": (
                "Retrieval augmented generation "
                "combines retrieval with generation."
            ),
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "doc-001",
            "document_name": "rag.pdf",
            "page_number": 2,
            "chunk_index": 1,
            "text": (
                "RAG can reduce hallucinations "
                "by grounding responses in retrieved context."
            ),
        },
        {
            "chunk_id": "chunk-003",
            "document_id": "doc-002",
            "document_name": "bert.pdf",
            "page_number": 3,
            "chunk_index": 0,
            "text": (
                "BERT is a transformer model "
                "used for natural language processing."
            ),
        },
    ]

    embeddings = [
        [0.1] * 768,
        [0.2] * 768,
        [0.3] * 768,
    ]

    client = create_qdrant_client()

    create_collection(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    insert_chunks(
        client=client,
        collection_name=COLLECTION_NAME,
        chunks=chunks,
        embeddings=embeddings,
    )

    keyword_search = KeywordSearch(chunks)

    results = hybrid_search(
        client=client,
        embedder=FakeEmbedder(),
        collection_name=COLLECTION_NAME,
        keyword_search=keyword_search,
        query="RAG hallucinations",
        dense_top_k=3,
        keyword_top_k=3,
        rrf_top_k=3,
    )

    assert len(results) == 3

    assert all(
        "chunk_id" in result
        for result in results
    )

    assert all(
        "rrf_score" in result
        for result in results
    )


@pytest.mark.parametrize(
    "query, dense_top_k, keyword_top_k, rrf_top_k, expected_message",
    [
        ("   ", 3, 3, 3, "Query cannot be empty."),
        ("RAG", 0, 3, 3, "dense_top_k must be greater than zero."),
        ("RAG", 3, 0, 3, "keyword_top_k must be greater than zero."),
        ("RAG", 3, 3, 0, "rrf_top_k must be greater than zero."),
    ],
)
def test_hybrid_search_invalid_arguments(
    query,
    dense_top_k,
    keyword_top_k,
    rrf_top_k,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        hybrid_search(
            client=None,
            embedder=None,
            collection_name="test_hybrid",
            keyword_search=None,
            query=query,
            dense_top_k=dense_top_k,
            keyword_top_k=keyword_top_k,
            rrf_top_k=rrf_top_k,
        )


class FakeKeywordSearch:
    def __init__(self, results):
        self.results = results

    def search(self, query: str, top_k: int = 10):
        return self.results[:top_k]


def make_result(chunk_id: str, text: str = "test chunk"):
    return {
        "chunk_id": chunk_id,
        "document_id": "doc-001",
        "document_name": "test.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "text": text,
    }


def run_hybrid_with_mocked_retrieval(dense_results, keyword_results):
    with (
        patch(
            "app.retrieval.hybrid_retrieval.search_similar_chunks",
            return_value=["mocked dense response"],
        ),
        patch(
            "app.retrieval.hybrid_retrieval.format_search_results",
            return_value=dense_results,
        ),
    ):
        return hybrid_search(
            client=None,
            embedder=FakeEmbedder(),
            collection_name="test_hybrid",
            keyword_search=FakeKeywordSearch(keyword_results),
            query="RAG",
            dense_top_k=3,
            keyword_top_k=3,
            rrf_top_k=10,
        )


def test_hybrid_search_with_empty_dense_results():
    keyword_results = [
        make_result("keyword-001"),
        make_result("keyword-002"),
    ]

    results = run_hybrid_with_mocked_retrieval(
        dense_results=[],
        keyword_results=keyword_results,
    )

    result_ids = [result["chunk_id"] for result in results]

    assert set(result_ids) == {"keyword-001", "keyword-002"}


def test_hybrid_search_with_empty_keyword_results():
    dense_results = [
        make_result("dense-001"),
        make_result("dense-002"),
    ]

    results = run_hybrid_with_mocked_retrieval(
        dense_results=dense_results,
        keyword_results=[],
    )

    result_ids = [result["chunk_id"] for result in results]

    assert set(result_ids) == {"dense-001", "dense-002"}


def test_hybrid_search_fuses_overlapping_chunks():
    shared = make_result("shared-001")
    dense_only = make_result("dense-only-001")
    keyword_only = make_result("keyword-only-001")

    results = run_hybrid_with_mocked_retrieval(
        dense_results=[shared, dense_only],
        keyword_results=[shared, keyword_only],
    )

    result_ids = [result["chunk_id"] for result in results]

    assert result_ids.count("shared-001") == 1
    assert set(result_ids) == {
        "shared-001",
        "dense-only-001",
        "keyword-only-001",
    }
    assert results[0]["chunk_id"] == "shared-001"
    assert results[0]["rrf_score"] > results[1]["rrf_score"]
