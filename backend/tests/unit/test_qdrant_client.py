from unittest.mock import Mock
import pytest
from app.database.qdrant_client import insert_chunk
from app.database.qdrant_client import insert_chunks
from uuid import uuid5, NAMESPACE_URL

from uuid import NAMESPACE_URL, uuid5

from app.database.qdrant_client import (
    create_collection,
    create_qdrant_client,
    insert_chunk,
)

def test_insert_chunk():
    client = create_qdrant_client()
    collection_name = "research_documents"

    create_collection(client, collection_name)

    chunk = {
        "chunk_id": "chunk-001",
        "document_id": "document-001",
        "document_name": "research_paper.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "text": "Neural networks learn by adjusting weights.",
    }

    embedding = [0.1] * 768

    insert_chunk(
        client=client,
        collection_name=collection_name,
        chunk=chunk,
        embedding=embedding,
    )

    results = client.retrieve(
        collection_name=collection_name,
        ids=[
            str(uuid5(NAMESPACE_URL, chunk["chunk_id"]))
        ],
        with_payload=True,
    )

    assert len(results) == 1
    assert results[0].payload["text"] == chunk["text"]


def test_insert_chunks():
    client = create_qdrant_client()
    collection_name = "batch_chunks"

    create_collection(client, collection_name)

    chunks = [
        {
            "chunk_id": "chunk-001",
            "document_id": "document-001",
            "document_name": "paper.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "Machine learning uses data.",
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "document-001",
            "document_name": "paper.pdf",
            "page_number": 1,
            "chunk_index": 1,
            "text": "Neural networks learn patterns.",
        },
    ]

    embeddings = [
        [0.1] * 768,
        [0.2] * 768,
    ]

    insert_chunks(
        client=client,
        collection_name=collection_name,
        chunks=chunks,
        embeddings=embeddings,
    )

    stored_count = client.count(
        collection_name=collection_name,
        exact=True,
    ).count

    assert stored_count == 2


def test_create_collection_skips_existing_collection():
    client = Mock()
    client.collection_exists.return_value = True

    create_collection(client, "research_documents")

    client.collection_exists.assert_called_once_with(
        "research_documents"
    )
    client.create_collection.assert_not_called()


def test_insert_chunks_rejects_mismatched_lengths():
    client = Mock()

    chunks = [
        {
            "chunk_id": "chunk-001",
            "document_id": "document-001",
            "document_name": "paper.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "Sample text",
        }
    ]

    embeddings = []

    with pytest.raises(
        ValueError,
        match="Number of chunks must match number of embeddings",
    ):
        insert_chunks(
            client=client,
            collection_name="research_documents",
            chunks=chunks,
            embeddings=embeddings,
        )

    client.upsert.assert_not_called()


def test_insert_chunks_skips_upsert_for_empty_lists():
    client = Mock()

    insert_chunks(
        client=client,
        collection_name="research_documents",
        chunks=[],
        embeddings=[],
    )

    client.upsert.assert_not_called()
