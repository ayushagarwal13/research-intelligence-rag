from app.retrieval.query_rewriter import QueryRewriter


class FakeLLMService:
    def __init__(self, response: str):
        self.response = response
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


def test_query_rewriter_returns_original_query_without_history():
    llm_service = FakeLLMService(
        response="unused"
    )

    rewriter = QueryRewriter(
        llm_service=llm_service
    )

    result = rewriter.rewrite(
        query="What is regularization?",
        conversation_history=[],
    )

    assert result == "What is regularization?"
    assert llm_service.last_prompt is None


def test_query_rewriter_rewrites_follow_up_question():
    llm_service = FakeLLMService(
        response=(
            "Why is regularization necessary "
            "for inverse problems?"
        )
    )

    rewriter = QueryRewriter(
        llm_service=llm_service
    )

    history = [
        {
            "role": "user",
            "content": "What is regularization?",
        },
        {
            "role": "assistant",
            "content": (
                "Regularization approximates "
                "a generalized inverse."
            ),
        },
    ]

    result = rewriter.rewrite(
        query="Why is it necessary for inverse problems?",
        conversation_history=history,
    )

    assert (
        result
        == "Why is regularization necessary "
        "for inverse problems?"
    )

    assert llm_service.last_prompt is not None
    assert "What is regularization?" in llm_service.last_prompt
    assert "Why is it necessary for inverse problems?" in llm_service.last_prompt


def test_query_rewriter_requires_llm_for_conversation():
    rewriter = QueryRewriter()

    history = [
        {
            "role": "user",
            "content": "What is regularization?",
        }
    ]

    try:
        rewriter.rewrite(
            query="Why is it necessary?",
            conversation_history=history,
        )
        assert False, "Expected ValueError"
    except ValueError as error:
        assert (
            str(error)
            == "LLM service is required when "
            "conversation history is provided."
        )


def test_query_rewriter_rejects_empty_query():
    rewriter = QueryRewriter()

    try:
        rewriter.rewrite(query="   ")
        assert False, "Expected ValueError"
    except ValueError as error:
        assert str(error) == "Query cannot be empty."
