from unittest.mock import Mock

import pytest

from app.services.query_service import QueryService


def test_query_service_calls_rag_service():
    rag_service = Mock()

    rag_service.answer.return_value = {
        "answer": "RAG retrieves relevant information.",
        "sources": [],
    }

    service = QueryService(
        rag_service=rag_service,
    )

    result = service.answer_query(
        query="What is RAG?"
    )

    assert result == {
        "answer": "RAG retrieves relevant information.",
        "sources": [],
    }

    rag_service.answer.assert_called_once_with(
        query="What is RAG?",
        conversation_history=None,
    )


def test_query_service_passes_conversation_history():
    rag_service = Mock()

    rag_service.answer.return_value = {
        "answer": "RAG uses retrieved context.",
        "sources": [],
    }

    service = QueryService(
        rag_service=rag_service,
    )

    conversation_history = [
        {
            "role": "user",
            "content": "What is RAG?",
        },
        {
            "role": "assistant",
            "content": "RAG retrieves context.",
        },
    ]

    result = service.answer_query(
        query="How does it work?",
        conversation_history=conversation_history,
    )

    assert result["answer"] == (
        "RAG uses retrieved context."
    )

    rag_service.answer.assert_called_once_with(
        query="How does it work?",
        conversation_history=conversation_history,
    )


def test_query_service_rejects_empty_query():
    rag_service = Mock()

    service = QueryService(
        rag_service=rag_service,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        service.answer_query("")


def test_query_service_rejects_whitespace_query():
    rag_service = Mock()

    service = QueryService(
        rag_service=rag_service,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        service.answer_query("   ")


def test_query_service_refreshes_keyword_index_on_first_query():
    rag_service = Mock()

    rag_service.answer.return_value = {
        "answer": "RAG retrieves relevant information.",
        "sources": [],
    }

    refresh_callback = Mock()

    service = QueryService(
        rag_service=rag_service,
        keyword_index_refresh_callback=refresh_callback,
        keyword_index_refresh_interval_seconds=30,
    )

    service.answer_query(
        query="What is RAG?"
    )

    refresh_callback.assert_called_once_with()


def test_query_service_does_not_refresh_within_interval(
    monkeypatch,
):
    rag_service = Mock()

    rag_service.answer.return_value = {
        "answer": "RAG retrieves relevant information.",
        "sources": [],
    }

    refresh_callback = Mock()

    service = QueryService(
        rag_service=rag_service,
        keyword_index_refresh_callback=refresh_callback,
        keyword_index_refresh_interval_seconds=30,
    )

    current_time = 1000.0

    monkeypatch.setattr(
        "app.services.query_service.time.monotonic",
        lambda: current_time,
    )

    service.answer_query(
        query="What is RAG?"
    )

    service.answer_query(
        query="What is hybrid retrieval?"
    )

    assert refresh_callback.call_count == 1


def test_query_service_refreshes_after_interval(
    monkeypatch,
):
    rag_service = Mock()

    rag_service.answer.return_value = {
        "answer": "RAG retrieves relevant information.",
        "sources": [],
    }

    refresh_callback = Mock()

    service = QueryService(
        rag_service=rag_service,
        keyword_index_refresh_callback=refresh_callback,
        keyword_index_refresh_interval_seconds=30,
    )

    current_time = 1000.0

    monkeypatch.setattr(
        "app.services.query_service.time.monotonic",
        lambda: current_time,
    )

    service.answer_query(
        query="What is RAG?"
    )

    assert refresh_callback.call_count == 1

    current_time = 1031.0

    service.answer_query(
        query="What is hybrid retrieval?"
    )

    assert refresh_callback.call_count == 2
