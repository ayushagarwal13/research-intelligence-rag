import os

import pytest

from app.services.gemini_service import GeminiService


@pytest.mark.integration
def test_real_gemini_generation():

    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip(
            "GEMINI_API_KEY is not configured."
        )

    service = GeminiService()

    response = service.generate(
        "In one sentence, explain what "
        "Retrieval-Augmented Generation is."
    )

    assert response
    assert isinstance(response, str)
