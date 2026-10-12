from app.services.retrieval_service import (
    retrieve_relevant_chunks,
)


class FakeEmbedder:
    def embed_text(self, text: str) -> list[float]:
        return [0.1] * 768


class FakeReranker:
    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int,
    ) -> list[dict]:
        return chunks[:top_k]


class FakeKeywordSearch:
    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict]:
        return [
            {
                "chunk_id": "chunk-001",
                "document_id": "doc-001",
                "document_name": "rag.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "text": "RAG retrieves external context.",
            },
            {
                "chunk_id": "chunk-002",
                "document_id": "doc-001",
                "document_name": "rag.pdf",
                "page_number": 2,
                "chunk_index": 1,
                "text": "RAG can reduce hallucinations.",
            },
        ][:top_k]


class FakeQueryRewriter:
    def rewrite(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        return "rewritten RAG query"


class FakeClient:
    def query_points(
        self,
        collection_name,
        query,
        query_filter,
        limit,
        with_payload,
    ):
        """
        Return fake Qdrant search results.
        """

        class FakeResult:
            def __init__(self, payload, score):
                self.payload = payload
                self.score = score

        return type(
            "FakeSearchResponse",
            (),
            {
                "points": [
                    FakeResult(
                        {
                            "chunk_id": "chunk-001",
                            "document_id": "doc-001",
                            "document_name": "rag.pdf",
                            "page_number": 1,
                            "chunk_index": 0,
                            "text": (
                                "RAG retrieves "
                                "external context."
                            ),
                        },
                        0.95,
                    ),
                    FakeResult(
                        {
                            "chunk_id": "chunk-002",
                            "document_id": "doc-001",
                            "document_name": "rag.pdf",
                            "page_number": 2,
                            "chunk_index": 1,
                            "text": (
                                "RAG can reduce "
                                "hallucinations."
                            ),
                        },
                        0.85,
                    ),
                ][:limit]
            },
        )()


def test_retrieve_relevant_chunks():
    """
    Verify that the retrieval service orchestrates
    hybrid retrieval and reranking.
    """

    results = retrieve_relevant_chunks(
        client=FakeClient(),
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        collection_name="test_collection",
        keyword_search=FakeKeywordSearch(),
        query="How does RAG work?",
        dense_top_k=2,
        keyword_top_k=2,
        rrf_top_k=2,
        final_top_k=1,
    )

    assert len(results) == 1

    assert "chunk_id" in results[0]
    assert "text" in results[0]


def test_empty_query_raises_error():
    try:
        retrieve_relevant_chunks(
            client=FakeClient(),
            embedder=FakeEmbedder(),
            reranker=FakeReranker(),
            collection_name="test_collection",
            keyword_search=FakeKeywordSearch(),
            query="",
        )

        assert False, "Expected ValueError"

    except ValueError:
        pass


def test_final_top_k_must_be_positive():
    try:
        retrieve_relevant_chunks(
            client=FakeClient(),
            embedder=FakeEmbedder(),
            reranker=FakeReranker(),
            collection_name="test_collection",
            keyword_search=FakeKeywordSearch(),
            query="What is RAG?",
            final_top_k=0,
        )

        assert False, "Expected ValueError"

    except ValueError:
        pass


class FakeReranker:

    def __init__(self):
        self.received_query = None

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int,
    ) -> list[dict]:

        self.received_query = query

        return chunks[:top_k]


def test_retrieve_relevant_chunks_uses_query_rewriter():

    reranker = FakeReranker()

    results = retrieve_relevant_chunks(
        client=FakeClient(),
        embedder=FakeEmbedder(),
        reranker=reranker,
        collection_name="test_collection",
        keyword_search=FakeKeywordSearch(),
        query="How does it work?",
        query_rewriter=FakeQueryRewriter(),
        conversation_history=[
            {
                "role": "user",
                "content": "What is RAG?",
            }
        ],
        dense_top_k=2,
        keyword_top_k=2,
        rrf_top_k=2,
        final_top_k=1,
    )

    assert len(results) == 1

    assert (
        reranker.received_query
        == "rewritten RAG query"
    )
