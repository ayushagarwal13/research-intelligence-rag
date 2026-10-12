import pytest

from google.genai import errors

from app.services.gemini_service import GeminiService
from app.services.llm_service import LLMServiceUnavailableError


class FakeResponse:
    def __init__(self, text: str | None):
        self.text = text


class FakeModels:
    def __init__(self, responses=None, exceptions=None):
        self.responses = responses or []
        self.exceptions = exceptions or []
        self.call_count = 0

    def generate_content(
        self,
        model: str,
        contents: str,
    ):
        self.call_count += 1

        if self.exceptions:
            exception = self.exceptions.pop(0)
            raise exception

        return self.responses.pop(0)


class FakeClient:
    def __init__(self, models):
        self.models = models


def create_service(
    monkeypatch,
    fake_models,
    max_retries=2,
    retry_delay=0,
):
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-api-key",
    )

    service = GeminiService(
        model_name="test-model",
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    service.client = FakeClient(
        models=fake_models
    )

    return service


def test_gemini_service_requires_api_key(
    monkeypatch,
):
    monkeypatch.delenv(
        "GEMINI_API_KEY",
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="GEMINI_API_KEY",
    ):
        GeminiService()


def test_gemini_service_generate(
    monkeypatch,
):
    fake_models = FakeModels(
        responses=[
            FakeResponse(
                "This is a generated answer."
            )
        ]
    )

    service = create_service(
        monkeypatch,
        fake_models,
    )

    result = service.generate(
        "Explain RAG."
    )

    assert result == (
        "This is a generated answer."
    )

    assert fake_models.call_count == 1


def test_gemini_service_rejects_empty_prompt(
    monkeypatch,
):
    fake_models = FakeModels()

    service = create_service(
        monkeypatch,
        fake_models,
    )

    with pytest.raises(
        ValueError,
        match="Prompt cannot be empty",
    ):
        service.generate("")


def test_gemini_service_rejects_whitespace_prompt(
    monkeypatch,
):
    fake_models = FakeModels()

    service = create_service(
        monkeypatch,
        fake_models,
    )

    with pytest.raises(
        ValueError,
        match="Prompt cannot be empty",
    ):
        service.generate("   ")


def test_gemini_service_retries_server_error(
    monkeypatch,
):
    server_error = errors.ServerError(
        503,
        {
            "error": {
                "code": 503,
                "message": "Temporary failure",
                "status": "UNAVAILABLE",
            }
        },
    )

    fake_models = FakeModels(
        responses=[
            FakeResponse(
                "Success after retry."
            )
        ],
        exceptions=[
            server_error,
        ],
    )

    service = create_service(
        monkeypatch,
        fake_models,
    )

    result = service.generate(
        "Explain hybrid retrieval."
    )

    assert result == "Success after retry."
    assert fake_models.call_count == 2


def test_gemini_service_stops_after_max_retries(
    monkeypatch,
):
    errors_to_raise = [
        errors.ServerError(
            503,
            {
                "error": {
                    "code": 503,
                    "message": "Temporary failure",
                    "status": "UNAVAILABLE",
                }
            },
        )
        for _ in range(3)
    ]

    fake_models = FakeModels(
        exceptions=errors_to_raise,
    )

    service = create_service(
        monkeypatch,
        fake_models,
        max_retries=2,
    )

    with pytest.raises(
        LLMServiceUnavailableError,
        match="temporarily unavailable",
    ):
        service.generate(
            "Explain reranking."
        )

    assert fake_models.call_count == 3


def test_gemini_service_rejects_empty_response(
    monkeypatch,
):
    fake_models = FakeModels(
        responses=[
            FakeResponse(None)
        ]
    )

    service = create_service(
        monkeypatch,
        fake_models,
    )

    with pytest.raises(
        ValueError,
        match="Gemini returned an empty response",
    ):
        service.generate(
            "Explain RAG."
        )

    assert fake_models.call_count == 1


def test_gemini_service_does_not_retry_non_retryable_error(
    monkeypatch,
):
    fake_models = FakeModels(
        exceptions=[
            RuntimeError("Unexpected provider failure")
        ]
    )

    service = create_service(
        monkeypatch,
        fake_models,
    )

    with pytest.raises(
        RuntimeError,
        match="Unexpected provider failure",
    ):
        service.generate(
            "Explain embeddings."
        )

    assert fake_models.call_count == 1


def test_gemini_service_uses_exponential_backoff(
    monkeypatch,
):
    server_errors = [
        errors.ServerError(
            503,
            {
                "error": {
                    "code": 503,
                    "message": "Temporary failure",
                    "status": "UNAVAILABLE",
                }
            },
        )
        for _ in range(2)
    ]

    fake_models = FakeModels(
        responses=[
            FakeResponse("Success after retries.")
        ],
        exceptions=server_errors,
    )

    service = create_service(
        monkeypatch,
        fake_models,
        max_retries=2,
        retry_delay=1,
    )

    sleep_delays = []

    monkeypatch.setattr(
        "app.services.gemini_service.time.sleep",
        lambda delay: sleep_delays.append(delay),
    )

    result = service.generate(
        "Explain reciprocal rank fusion."
    )

    assert result == "Success after retries."
    assert fake_models.call_count == 3
    assert sleep_delays == [1, 2]
