from app.generation.context_builder import build_context
from app.generation.prompt_builder import build_prompt
from app.services.llm_service import LLMService


class GenerationService:
    """
    Orchestrates context construction, prompt construction,
    and LLM-based answer generation.
    """

    def __init__(
        self,
        llm_service: LLMService,
    ):
        self.llm_service = llm_service

    def generate_answer(
        self,
        query: str,
        chunks: list[dict],
    ) -> str:
        """
        Generate an answer using retrieved chunks.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not chunks:
            raise ValueError(
                "Retrieved chunks cannot be empty."
            )

        context = build_context(chunks)

        if not context:
            raise ValueError(
                "No usable context was found."
            )

        prompt = build_prompt(
            query=query,
            context=context,
        )

        answer = self.llm_service.generate(
            prompt
        )

        if not answer.strip():
            raise ValueError(
                "LLM returned an empty answer."
            )

        return answer.strip()
