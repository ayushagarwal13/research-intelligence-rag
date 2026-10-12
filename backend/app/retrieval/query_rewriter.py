from app.services.llm_service import LLMService


class QueryRewriter:
    """
    Prepare user queries for retrieval.
    """

    def __init__(
        self,
        llm_service: LLMService | None = None,
    ):
        self.llm_service = llm_service

    def rewrite(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """
        Convert a conversational query into a
        retrieval-ready standalone query when
        conversation history is available.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query = query.strip()

        # No conversation context means the query
        # is already treated as standalone.
        if not conversation_history:
            return query

        # An LLM is required when conversation context
        # needs to be incorporated.
        if self.llm_service is None:
            raise ValueError(
                "LLM service is required when "
                "conversation history is provided."
            )

        prompt = self._build_rewrite_prompt(
            query=query,
            conversation_history=conversation_history,
        )

        rewritten_query = self.llm_service.generate(
            prompt
        )

        if not rewritten_query.strip():
            raise ValueError(
                "LLM returned an empty rewritten query."
            )

        return rewritten_query.strip()

    @staticmethod
    def _build_rewrite_prompt(
        query: str,
        conversation_history: list[dict],
    ) -> str:
        """
        Build the prompt used to rewrite a
        conversational query.
        """

        history_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in conversation_history
        )

        return f"""
Rewrite the user's latest question into a
standalone search query.

Use the conversation history only to resolve
references such as "it", "they", "this", or
"the previous method".

Do not answer the question.
Return only the rewritten search query.

Conversation history:
{history_text}

Latest user question:
{query}
""".strip()
