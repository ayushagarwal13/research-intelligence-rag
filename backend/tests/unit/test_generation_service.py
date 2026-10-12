from unittest.mock import Mock

import pytest

from app.services.generation_service import (
    GenerationService,
)


def test_generate_answer_builds_prompt_and_calls_llm(
    monkeypatch,
):
    llm_service = Mock()

    llm_service.generate.return_value = (
        "RAG combines retrieval with generation."
    )

    service = GenerationService(
        llm_service=llm_service
    )

    chunks = [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "document_name": "paper.pdf",
            "page_number": 3,
            "text": (
                "Retrieval-Augmented Generation "
                "combines retrieval with generation."
            ),
        }
    ]

    monkeypatch.setattr(
        "app.services.generation_service.build_context",
        lambda chunks: (
            "Source: paper.pdf, Page: 3\n"
            "Retrieval-Augmented Generation "
            "combines retrieval with generation."
        ),
    )

    monkeypatch.setattr(
        "app.services.generation_service.build_prompt",
        lambda query, context: (
            f"QUESTION: {query}\n"
            f"CONTEXT: {context}"
        ),
    )

    result = service.generate_answer(
        query="What is RAG?",
        chunks=chunks,
    )

    assert result == (
        "RAG combines retrieval with generation."
    )

    llm_service.generate.assert_called_once_with(
        "QUESTION: What is RAG?\n"
        "CONTEXT: Source: paper.pdf, Page: 3\n"
        "Retrieval-Augmented Generation "
        "combines retrieval with generation."
    )


def test_generate_answer_rejects_empty_query():
    llm_service = Mock()

    service = GenerationService(
        llm_service=llm_service
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        service.generate_answer(
            query="",
            chunks=[
                {
                    "text": "some context"
                }
            ],
        )

    llm_service.generate.assert_not_called()


def test_generate_answer_rejects_empty_chunks():
    llm_service = Mock()

    service = GenerationService(
        llm_service=llm_service
    )

    with pytest.raises(
        ValueError,
        match="Retrieved chunks cannot be empty",
    ):
        service.generate_answer(
            query="What is RAG?",
            chunks=[],
        )

    llm_service.generate.assert_not_called()


def test_generate_answer_rejects_empty_llm_response(
    monkeypatch,
):
    llm_service = Mock()

    llm_service.generate.return_value = ""

    service = GenerationService(
        llm_service=llm_service
    )

    monkeypatch.setattr(
        "app.services.generation_service.build_context",
        lambda chunks: "valid context",
    )

    monkeypatch.setattr(
        "app.services.generation_service.build_prompt",
        lambda query, context: "valid prompt",
    )

    with pytest.raises(
        ValueError,
        match="LLM returned an empty answer",
    ):
        service.generate_answer(
            query="What is RAG?",
            chunks=[
                {
                    "text": "valid context"
                }
            ],
        )
