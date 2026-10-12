from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_query_service
from app.api.query import router


def create_test_app():
    app = FastAPI()
    app.include_router(router)

    return app


def test_query_endpoint_returns_answer():
    query_service = Mock()

    query_service.answer_query.return_value = {
        "answer": "RAG retrieves relevant context.",
        "sources": [],
    }

    app = create_test_app()

    app.dependency_overrides[
        get_query_service
    ] = lambda: query_service

    client = TestClient(app)

    response = client.post(
        "/query",
        json={
            "query": "What is RAG?"
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "answer": "RAG retrieves relevant context.",
        "sources": [],
    }

    query_service.answer_query.assert_called_once_with(
        query="What is RAG?",
        conversation_history=None,
    )


def test_query_endpoint_accepts_conversation_history():
    query_service = Mock()

    query_service.answer_query.return_value = {
        "answer": "It retrieves relevant context.",
        "sources": [],
    }

    app = create_test_app()

    app.dependency_overrides[
        get_query_service
    ] = lambda: query_service

    client = TestClient(app)

    history = [
        {
            "role": "user",
            "content": "What is RAG?",
        }
    ]

    response = client.post(
        "/query",
        json={
            "query": "How does it work?",
            "conversation_history": history,
        },
    )

    assert response.status_code == 200

    query_service.answer_query.assert_called_once_with(
        query="How does it work?",
        conversation_history=history,
    )


def test_query_endpoint_returns_400_for_service_error():
    query_service = Mock()

    query_service.answer_query.side_effect = ValueError(
        "Query cannot be empty."
    )

    app = create_test_app()

    app.dependency_overrides[
        get_query_service
    ] = lambda: query_service

    client = TestClient(app)

    response = client.post(
        "/query",
        json={
            "query": "test"
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Query cannot be empty."
    }
