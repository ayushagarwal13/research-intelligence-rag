from app.database.chunk_repository import ChunkRepository
from app.database.qdrant_client import (
    create_collection,
    create_qdrant_client,
)
from app.database.session import SessionLocal
from app.embeddings.embedder import BGEEmbedder
from app.reranking.reranker import CrossEncoderReranker
from app.retrieval.keyword_search import KeywordSearch
from app.retrieval.query_rewriter import QueryRewriter
from app.services.groq_service import GroqService
from app.services.generation_service import GenerationService
from app.services.query_service import QueryService
from app.services.rag_service import RAGService


COLLECTION_NAME = "research_documents"

# BM25 is refreshed from PostgreSQL at most once during
# this interval for the running API process.
KEYWORD_INDEX_REFRESH_INTERVAL_SECONDS = 30.0


class ApplicationContainer:
    """
    Holds the long-lived dependencies shared by the
    query service and retrieval pipeline.
    """

    def __init__(
        self,
        chunks: list[dict] | None = None,
    ):
        self.qdrant_client = create_qdrant_client()

        create_collection(
            client=self.qdrant_client,
            collection_name=COLLECTION_NAME,
        )

        self.embedder = BGEEmbedder()

        self.reranker = CrossEncoderReranker()

        keyword_chunks = chunks

        if keyword_chunks is None:
            keyword_chunks = self._load_chunks_from_database()

        self.keyword_search = KeywordSearch(
            [{"text": ""}]
        )

        self.keyword_search.update_chunks(
            keyword_chunks
        )

        self.groq_service = GroqService()

        self.query_rewriter = QueryRewriter(
            llm_service=self.groq_service
        )

        self.generation_service = GenerationService(
            llm_service=self.groq_service
        )

        self.rag_service = RAGService(
            client=self.qdrant_client,
            embedder=self.embedder,
            reranker=self.reranker,
            collection_name=COLLECTION_NAME,
            keyword_search=self.keyword_search,
            generation_service=self.generation_service,
            query_rewriter=self.query_rewriter,
            dense_top_k=5,
            keyword_top_k=5,
            rrf_top_k=10,
            final_top_k=3,
        )

        self.query_service = QueryService(
            rag_service=self.rag_service,
            keyword_index_refresh_callback=(
                self.refresh_keyword_index
            ),
            keyword_index_refresh_interval_seconds=(
                KEYWORD_INDEX_REFRESH_INTERVAL_SECONDS
            ),
        )

    def refresh_keyword_index(self) -> None:
        """
        Rebuild the in-memory BM25 index from the
        persistent chunks stored in PostgreSQL.
        """
        keyword_chunks = self._load_chunks_from_database()

        self.keyword_search.update_chunks(
            keyword_chunks
        )

    @staticmethod
    def _load_chunks_from_database() -> list[dict]:
        """
        Load all persisted chunks from the database so the
        BM25 keyword index can be reconstructed after an
        application restart or refreshed during runtime.
        """
        with SessionLocal() as session:
            repository = ChunkRepository(
                session=session
            )

            return repository.get_all_chunks()
