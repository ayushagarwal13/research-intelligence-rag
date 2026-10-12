from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from google.genai import errors

from app.api.query import query
from app.schemas.query import QueryRequest
from app.services.gemini_service import GeminiService
from app.services.llm_service import (
    LLMServiceUnavailableError,
)


def _create_gemini_service(
    monkeypatch,
    fake_client,
):
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-api-key",
    )

    monkeypatch.setattr(
        "app.services.gemini_service.genai.Client",
        lambda api_key: fake_client,
    )

    return GeminiService(
        max_retries=0,
        retry_delay=0,
    )


def test_gemini_429_becomes_service_unavailable(
    monkeypatch,
):
    fake_client = Mock()

    fake_client.models.generate_content.side_effect = (
        errors.ClientError(
            429,
            {
                "error": {
                    "code": 429,
                    "message": "Quota exhausted.",
                    "status": "RESOURCE_EXHAUSTED",
                }
            },
            None,
        )
    )

    service = _create_gemini_service(
        monkeypatch,
        fake_client,
    )

    with pytest.raises(
        LLMServiceUnavailableError,
        match="rate limit or quota",
    ):
        service.generate("Test prompt")


def test_gemini_server_error_becomes_service_unavailable(
    monkeypatch,
):
    fake_client = Mock()

    fake_client.models.generate_content.side_effect = (
        errors.ServerError(
            503,
            {
                "error": {
                    "code": 503,
                    "message": "Service unavailable.",
                    "status": "UNAVAILABLE",
                }
            },
            None,
        )
    )

    service = _create_gemini_service(
        monkeypatch,
        fake_client,
    )

    with pytest.raises(
        LLMServiceUnavailableError,
        match="temporarily unavailable",
    ):
        service.generate("Test prompt")


def test_gemini_non_429_client_error_is_not_hidden(
    monkeypatch,
):
    fake_client = Mock()

    fake_client.models.generate_content.side_effect = (
        errors.ClientError(
            400,
            {
                "error": {
                    "code": 400,
                    "message": "Invalid request.",
                    "status": "INVALID_ARGUMENT",
                }
            },
            None,
        )
    )

    service = _create_gemini_service(
        monkeypatch,
        fake_client,
    )

    with pytest.raises(
        errors.ClientError
    ) as exc_info:
        service.generate("Test prompt")

    assert exc_info.value.code == 400


def test_query_maps_llm_unavailable_to_503():
    class FailingQueryService:
        def answer_query(
            self,
            query,
            conversation_history,
        ):
            raise LLMServiceUnavailableError(
                "LLM service unavailable."
            )

    request = QueryRequest(
        query="What is regularization?",
        conversation_history=[],
    )

    with pytest.raises(
        HTTPException
    ) as exc_info:
        query(
            request=request,
            query_service=FailingQueryService(),
        )

    assert exc_info.value.status_code == 503
    assert (
        exc_info.value.detail
        == "LLM service unavailable."
    )
