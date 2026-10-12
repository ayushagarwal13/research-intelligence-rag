import time
from collections.abc import Callable

from app.services.rag_service import RAGService


class QueryService:
    """
    Application service for processing research queries.

    This layer keeps the API independent from the
    internal RAG orchestration.

    The optional keyword-index refresh callback allows
    the API process to refresh its in-memory BM25 index
    from the persistent database after the ingestion
    worker adds new document chunks.
    """

    def __init__(
        self,
        rag_service: RAGService,
        keyword_index_refresh_callback: (
            Callable[[], None] | None
        ) = None,
        keyword_index_refresh_interval_seconds: float = 30.0,
    ):
        if keyword_index_refresh_interval_seconds < 0:
            raise ValueError(
                "Keyword index refresh interval cannot be negative."
            )

        self.rag_service = rag_service
        self.keyword_index_refresh_callback = (
            keyword_index_refresh_callback
        )
        self.keyword_index_refresh_interval_seconds = (
            keyword_index_refresh_interval_seconds
        )

        self._last_keyword_index_refresh = 0.0

    def answer_query(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        """
        Process a research question through the RAG pipeline.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        self._refresh_keyword_index_if_needed()

        return self.rag_service.answer(
            query=query,
            conversation_history=conversation_history,
        )

    def _refresh_keyword_index_if_needed(self) -> None:
        """
        Refresh the BM25 index when the configured refresh
        interval has elapsed.

        If no refresh callback was configured, this method
        does nothing. This keeps QueryService compatible
        with unit tests and alternative RAG configurations.
        """

        if self.keyword_index_refresh_callback is None:
            return

        current_time = time.monotonic()

        elapsed_time = (
            current_time - self._last_keyword_index_refresh
        )

        if (
            elapsed_time
            < self.keyword_index_refresh_interval_seconds
        ):
            return

        self.keyword_index_refresh_callback()

        self._last_keyword_index_refresh = current_time
