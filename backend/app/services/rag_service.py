from app.services.generation_service import GenerationService
from app.services.retrieval_service import retrieve_relevant_chunks


class RAGService:
    """
    Orchestrates retrieval and answer generation.
    """

    def __init__(
        self,
        client,
        embedder,
        reranker,
        collection_name: str,
        keyword_search,
        generation_service: GenerationService,
        query_rewriter=None,
        dense_top_k: int = 10,
        keyword_top_k: int = 10,
        rrf_top_k: int = 20,
        final_top_k: int = 5,
    ):
        self.client = client
        self.embedder = embedder
        self.reranker = reranker
        self.collection_name = collection_name
        self.keyword_search = keyword_search
        self.generation_service = generation_service
        self.query_rewriter = query_rewriter

        self.dense_top_k = dense_top_k
        self.keyword_top_k = keyword_top_k
        self.rrf_top_k = rrf_top_k
        self.final_top_k = final_top_k

    def answer(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        """
        Retrieve relevant research chunks and
        generate an answer from them.
        """
        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        retrieved_chunks = retrieve_relevant_chunks(
            client=self.client,
            embedder=self.embedder,
            reranker=self.reranker,
            collection_name=self.collection_name,
            keyword_search=self.keyword_search,
            query=query,
            query_rewriter=self.query_rewriter,
            conversation_history=conversation_history,
            dense_top_k=self.dense_top_k,
            keyword_top_k=self.keyword_top_k,
            rrf_top_k=self.rrf_top_k,
            final_top_k=self.final_top_k,
        )

        answer = self.generation_service.generate_answer(
            query=query,
            chunks=retrieved_chunks,
        )

        return {
            "answer": answer,
            "sources": retrieved_chunks,
        }
