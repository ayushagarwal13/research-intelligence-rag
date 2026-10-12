import pytest
from pydantic import ValidationError

from app.schemas.query import (
    QueryRequest,
    QueryResponse,
    QuerySource,
)


def test_query_request_accepts_valid_query():
    request = QueryRequest(
        query="What is retrieval-augmented generation?"
    )

    assert request.query == (
        "What is retrieval-augmented generation?"
    )

    assert request.conversation_history is None


def test_query_request_accepts_conversation_history():
    history = [
        {
            "role": "user",
            "content": "What is RAG?",
        },
        {
            "role": "assistant",
            "content": "RAG combines retrieval and generation.",
        },
    ]

    request = QueryRequest(
        query="How does it work?",
        conversation_history=history,
    )

    assert request.conversation_history == history


def test_query_request_rejects_empty_query():
    with pytest.raises(ValidationError):
        QueryRequest(query="")


def test_query_source_accepts_valid_source():
    source = QuerySource(
        chunk_id="chunk-001",
        document_id="doc-001",
        document_name="paper.pdf",
        page_number=3,
        text="Relevant research content.",
        rerank_score=2.5,
    )

    assert source.chunk_id == "chunk-001"
    assert source.page_number == 3
    assert source.rerank_score == 2.5


def test_query_response_accepts_valid_response():
    source = QuerySource(
        chunk_id="chunk-001",
        document_id="doc-001",
        document_name="paper.pdf",
        page_number=3,
        text="Relevant research content.",
        rerank_score=2.5,
    )

    response = QueryResponse(
        answer="This is the answer.",
        sources=[source],
    )

    assert response.answer == "This is the answer."
    assert len(response.sources) == 1
