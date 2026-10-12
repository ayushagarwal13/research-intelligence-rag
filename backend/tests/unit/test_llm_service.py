import pytest

from app.services.llm_service import LLMService


def test_llm_service_is_abstract():
    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class",
    ):
        LLMService()
