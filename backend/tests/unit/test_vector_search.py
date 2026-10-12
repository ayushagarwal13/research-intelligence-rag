from app.retrieval.vector_search import (
    format_search_results,
    search_similar_chunks,
)
from app.database.qdrant_client import create_collection
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient
import pytest



def create_test_qdrant_client() -> QdrantClient:
    """Create isolated in-memory Qdrant storage for unit tests."""
    return QdrantClient(":memory:")


def test_search_similar_chunks():
    client = create_test_qdrant_client()
    collection_name = "test_documents"

    try:
        create_collection(client, collection_name)

        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=1,
                    vector=[1.0] + [0.0] * 767,
                    payload={
                        "chunk_id": "chunk-001",
                        "document_id": "document-001",
                        "document_name": "paper.pdf",
                        "page_number": 1,
                        "chunk_index": 0,
                        "text": "Machine learning uses data.",
                    },
                ),
                PointStruct(
                    id=2,
                    vector=[0.0, 1.0] + [0.0] * 766,
                    payload={
                        "chunk_id": "chunk-002",
                        "document_id": "document-001",
                        "document_name": "paper.pdf",
                        "page_number": 2,
                        "chunk_index": 1,
                        "text": "Neural networks learn patterns.",
                    },
                ),
            ],
        )

        results = search_similar_chunks(
            client=client,
            collection_name=collection_name,
            query_embedding=[1.0] + [0.0] * 767,
            top_k=1,
        )

        assert len(results) == 1
        assert results[0].payload["chunk_id"] == "chunk-001"

    finally:
        client.close()


def test_format_search_results():
    class FakeResult:
        score = 0.87

        payload = {
            "chunk_id": "chunk-001",
            "document_id": "document-001",
            "document_name": "research_paper.pdf",
            "page_number": 3,
            "chunk_index": 2,
            "text": (
                "Retrieval augmented generation "
                "combines retrieval and generation."
            ),
        }

    results = format_search_results([FakeResult()])

    assert len(results) == 1

    result = results[0]

    assert result["chunk_id"] == "chunk-001"
    assert result["document_id"] == "document-001"
    assert result["document_name"] == "research_paper.pdf"
    assert result["page_number"] == 3
    assert result["chunk_index"] == 2
    assert result["text"].startswith("Retrieval augmented")
    assert result["similarity_score"] == 0.87


def test_search_with_document_filter():
    client = create_test_qdrant_client()
    collection_name = "filtered_documents"

    try:
        create_collection(client, collection_name)

        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=1,
                    vector=[1.0] + [0.0] * 767,
                    payload={
                        "chunk_id": "chunk-001",
                        "document_id": "document-001",
                        "document_name": "paper_one.pdf",
                        "page_number": 1,
                        "chunk_index": 0,
                        "text": "Content from document one.",
                    },
                ),
                PointStruct(
                    id=2,
                    vector=[1.0] + [0.0] * 767,
                    payload={
                        "chunk_id": "chunk-002",
                        "document_id": "document-002",
                        "document_name": "paper_two.pdf",
                        "page_number": 1,
                        "chunk_index": 0,
                        "text": "Content from document two.",
                    },
                ),
            ],
        )

        results = search_similar_chunks(
            client=client,
            collection_name=collection_name,
            query_embedding=[1.0] + [0.0] * 767,
            top_k=5,
            document_id="document-001",
        )

        assert len(results) == 1
        assert results[0].payload["document_id"] == "document-001"

    finally:
        client.close()


def test_search_with_page_filter():
    client = create_test_qdrant_client()
    collection_name = "page_filtered_documents"

    try:
        create_collection(client, collection_name)

        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=1,
                    vector=[1.0] + [0.0] * 767,
                    payload={
                        "chunk_id": "chunk-001",
                        "document_id": "document-001",
                        "document_name": "paper.pdf",
                        "page_number": 1,
                        "chunk_index": 0,
                        "text": "Page one content.",
                    },
                ),
                PointStruct(
                    id=2,
                    vector=[1.0] + [0.0] * 767,
                    payload={
                        "chunk_id": "chunk-002",
                        "document_id": "document-001",
                        "document_name": "paper.pdf",
                        "page_number": 3,
                        "chunk_index": 1,
                        "text": "Page three content.",
                    },
                ),
            ],
        )

        results = search_similar_chunks(
            client=client,
            collection_name=collection_name,
            query_embedding=[1.0] + [0.0] * 767,
            top_k=5,
            page_number=3,
        )

        assert len(results) == 1
        assert results[0].payload["page_number"] == 3

    finally:
        client.close()


def test_invalid_page_range_raises_error():
    client = create_test_qdrant_client()

    try:
        with pytest.raises(
            ValueError,
            match="^min_page cannot be greater than max_page\\.$",
        ):
            search_similar_chunks(
                client=client,
                collection_name="documents",
                query_embedding=[1.0] + [0.0] * 767,
                min_page=5,
                max_page=2,
            )
    finally:
        client.close()


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        (
            {"query_embedding": []},
            "Query embedding cannot be empty.",
        ),
        (
            {"top_k": 0},
            "top_k must be greater than zero.",
        ),
        (
            {"page_number": 0},
            "page_number must be greater than zero.",
        ),
        (
            {"min_page": 0},
            "min_page must be greater than zero.",
        ),
        (
            {"max_page": 0},
            "max_page must be greater than zero.",
        ),
    ],
)
def test_search_invalid_arguments(kwargs, expected_message):
    client = create_test_qdrant_client()

    arguments = {
        "client": client,
        "collection_name": "documents",
        "query_embedding": [1.0] + [0.0] * 767,
    }
    arguments.update(kwargs)

    try:
        with pytest.raises(
            ValueError,
            match=f"^{expected_message.replace('.', r'\.')}$",
        ):
            search_similar_chunks(**arguments)
    finally:
        client.close()


def test_format_search_results_with_empty_payload():
    class FakeResult:
        score = 0.5
        payload = None

    results = format_search_results([FakeResult()])

    assert results == [
        {
            "chunk_id": None,
            "document_id": None,
            "document_name": None,
            "page_number": None,
            "chunk_index": None,
            "text": None,
            "similarity_score": 0.5,
        }
    ]


def test_search_with_page_range_filter():
    client = create_test_qdrant_client()
    collection_name = "range_filtered_documents"

    try:
        create_collection(client, collection_name)

        points = []

        for page in [1, 2, 3, 4]:
            points.append(
                PointStruct(
                    id=page,
                    vector=[1.0] + [0.0] * 767,
                    payload={
                        "chunk_id": f"chunk-{page:03}",
                        "document_id": "document-001",
                        "document_name": "paper.pdf",
                        "page_number": page,
                        "chunk_index": page - 1,
                        "text": f"Content from page {page}.",
                    },
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
        )

        results = search_similar_chunks(
            client=client,
            collection_name=collection_name,
            query_embedding=[1.0] + [0.0] * 767,
            top_k=10,
            min_page=2,
            max_page=3,
        )

        result_pages = {
            result.payload["page_number"]
            for result in results
        }

        assert result_pages == {2, 3}

    finally:
        client.close()

