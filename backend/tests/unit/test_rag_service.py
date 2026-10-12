from unittest.mock import Mock

import pytest

from app.services.rag_service import RAGService


def create_rag_service():
    client = Mock()
    embedder = Mock()
    reranker = Mock()
    keyword_search = Mock()
    generation_service = Mock()
    query_rewriter = Mock()

    service = RAGService(
        client=client,
        embedder=embedder,
        reranker=reranker,
        collection_name="research_documents",
        keyword_search=keyword_search,
        generation_service=generation_service,
        query_rewriter=query_rewriter,
    )

    return (
        service,
        client,
        embedder,
        reranker,
        keyword_search,
        generation_service,
        query_rewriter,
    )


def test_answer_retrieves_chunks_and_generates_answer(
    monkeypatch,
):
    (
        service,
        client,
        embedder,
        reranker,
        keyword_search,
        generation_service,
        query_rewriter,
    ) = create_rag_service()

    retrieved_chunks = [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "document_name": "paper.pdf",
            "page_number": 3,
            "text": "RAG combines retrieval with generation.",
        }
    ]

    generation_service.generate_answer.return_value = (
        "RAG combines retrieval with generation."
    )

    retrieve_mock = Mock(
        return_value=retrieved_chunks
    )

    monkeypatch.setattr(
        "app.services.rag_service.retrieve_relevant_chunks",
        retrieve_mock,
    )

    result = service.answer(
        query="What is RAG?"
    )

    assert result["answer"] == (
        "RAG combines retrieval with generation."
    )

    assert result["sources"] == retrieved_chunks

    retrieve_mock.assert_called_once_with(
        client=client,
        embedder=embedder,
        reranker=reranker,
        collection_name="research_documents",
        keyword_search=keyword_search,
        query="What is RAG?",
        query_rewriter=query_rewriter,
        conversation_history=None,
        dense_top_k=10,
        keyword_top_k=10,
        rrf_top_k=20,
        final_top_k=5,
    )

    generation_service.generate_answer.assert_called_once_with(
        query="What is RAG?",
        chunks=retrieved_chunks,
    )


def test_answer_passes_conversation_history_to_retrieval(
    monkeypatch,
):
    (
        service,
        client,
        embedder,
        reranker,
        keyword_search,
        generation_service,
        query_rewriter,
    ) = create_rag_service()

    conversation_history = [
        {
            "role": "user",
            "content": "What is RAG?",
        },
        {
            "role": "assistant",
            "content": "RAG retrieves relevant context.",
        },
    ]

    retrieved_chunks = [
        {
            "chunk_id": "chunk-002",
            "document_id": "doc-002",
            "document_name": "rag.pdf",
            "page_number": 5,
            "text": "RAG retrieves relevant context.",
        }
    ]

    generation_service.generate_answer.return_value = (
        "RAG retrieves relevant context."
    )

    retrieve_mock = Mock(
        return_value=retrieved_chunks
    )

    monkeypatch.setattr(
        "app.services.rag_service.retrieve_relevant_chunks",
        retrieve_mock,
    )

    result = service.answer(
        query="How does it work?",
        conversation_history=conversation_history,
    )

    assert result["answer"] == (
        "RAG retrieves relevant context."
    )

    retrieve_mock.assert_called_once_with(
        client=client,
        embedder=embedder,
        reranker=reranker,
        collection_name="research_documents",
        keyword_search=keyword_search,
        query="How does it work?",
        query_rewriter=query_rewriter,
        conversation_history=conversation_history,
        dense_top_k=10,
        keyword_top_k=10,
        rrf_top_k=20,
        final_top_k=5,
    )

    generation_service.generate_answer.assert_called_once_with(
        query="How does it work?",
        chunks=retrieved_chunks,
    )


def test_answer_rejects_empty_query():
    (
        service,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = create_rag_service()

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        service.answer("")


def test_answer_rejects_whitespace_query():
    (
        service,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = create_rag_service()

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        service.answer("   ")


def test_rag_service_passes_retrieval_configuration(
    monkeypatch,
):
    captured = {}

    def fake_retrieve_relevant_chunks(**kwargs):
        captured.update(kwargs)

        return [
            {
                "text": "Relevant research content.",
                "document_name": "paper.pdf",
                "page_number": 1,
            }
        ]

    monkeypatch.setattr(
        "app.services.rag_service.retrieve_relevant_chunks",
        fake_retrieve_relevant_chunks,
    )

    class FakeGenerationService:
        def generate_answer(
            self,
            query,
            chunks,
        ):
            return "Generated answer."

    service = RAGService(
        client="client",
        embedder="embedder",
        reranker="reranker",
        collection_name="research_documents",
        keyword_search="keyword_search",
        generation_service=FakeGenerationService(),
        dense_top_k=7,
        keyword_top_k=8,
        rrf_top_k=12,
        final_top_k=3,
    )

    result = service.answer(
        query="What is this paper about?"
    )

    assert result["answer"] == "Generated answer."
    assert len(result["sources"]) == 1

    assert captured["dense_top_k"] == 7
    assert captured["keyword_top_k"] == 8
    assert captured["rrf_top_k"] == 12
    assert captured["final_top_k"] == 3
