import pytest

from app.generation.prompt_builder import build_prompt


def test_build_prompt_contains_query_and_context():

    query = "What is retrieval augmented generation?"

    context = (
        "[Source 1]\n"
        "Document: rag.pdf\n"
        "Page: 4\n"
        "Content:\n"
        "RAG retrieves external information."
    )

    prompt = build_prompt(
        query=query,
        context=context,
    )

    assert query in prompt
    assert context in prompt


def test_build_prompt_contains_source_instruction():

    prompt = build_prompt(
        query="What is RAG?",
        context="[Source 1]\nRAG retrieves information.",
    )

    assert "[Source 1]" in prompt
    assert "source" in prompt.lower()


def test_build_prompt_contains_grounding_instruction():

    prompt = build_prompt(
        query="What is RAG?",
        context="[Source 1]\nRAG retrieves information.",
    )

    assert "provided context" in prompt.lower()
    assert "invent facts" in prompt.lower()


def test_empty_query_raises_error():

    with pytest.raises(ValueError):

        build_prompt(
            query="",
            context="Some research context.",
        )


def test_whitespace_query_raises_error():

    with pytest.raises(ValueError):

        build_prompt(
            query="   ",
            context="Some research context.",
        )


def test_empty_context_raises_error():

    with pytest.raises(ValueError):

        build_prompt(
            query="What is RAG?",
            context="",
        )


def test_whitespace_context_raises_error():

    with pytest.raises(ValueError):

        build_prompt(
            query="What is RAG?",
            context="   ",
        )
