import os
from pathlib import Path

import pytest

from app.chunking.document_chunker import chunk_document
from app.database.qdrant_client import (
    create_collection,
    create_qdrant_client,
)
from app.embeddings.embedder import BGEEmbedder
from app.ingestion.pdf_loader import extract_text_from_pdf
from app.reranking.reranker import CrossEncoderReranker
from app.retrieval.keyword_search import KeywordSearch
from app.services.document_indexing_service import index_document
from app.services.gemini_service import GeminiService
from app.services.generation_service import GenerationService
from app.services.rag_service import RAGService
from app.services.retrieval_service import (
    retrieve_relevant_chunks,
)


@pytest.mark.integration
def test_full_rag_pipeline():
    """
    Verify the complete RAG pipeline:

    PDF
        -> extraction
        -> BGE embeddings
        -> Qdrant
        -> BM25
        -> RRF
        -> cross-encoder reranking
        -> context generation
        -> Gemini answer
    """

    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip(
            "GEMINI_API_KEY is not configured."
        )

    pdf_files = list(
        Path("data/documents").glob("*.pdf")
    )

    assert pdf_files, "No PDF files found."

    # --------------------------------------------------
    # 1. Extract one real research paper
    # --------------------------------------------------

    document = extract_text_from_pdf(
        str(pdf_files[0])
    )

    assert document["document_id"]
    assert document["pages"]

    # --------------------------------------------------
    # 2. Create local Qdrant collection
    # --------------------------------------------------

    client = create_qdrant_client()

    collection_name = "full_rag_test"

    create_collection(
        client=client,
        collection_name=collection_name,
    )

    # --------------------------------------------------
    # 3. Load real BGE embedding model
    # --------------------------------------------------

    embedder = BGEEmbedder()

    # --------------------------------------------------
    # 4. Index the research paper
    # --------------------------------------------------

    indexed_count = index_document(
        client=client,
        embedder=embedder,
        collection_name=collection_name,
        document=document,
        chunk_size=500,
        overlap=50,
    )

    assert indexed_count > 0

    # --------------------------------------------------
    # 5. Build BM25 index from the same chunks
    # --------------------------------------------------

    chunks = chunk_document(
        document=document,
        chunk_size=500,
        overlap=50,
    )

    keyword_search = KeywordSearch(
        chunks
    )

    # --------------------------------------------------
    # 6. Load real cross-encoder reranker
    # --------------------------------------------------

    reranker = CrossEncoderReranker()

    # --------------------------------------------------
    # 7. Test retrieval before generation
    # --------------------------------------------------

    query = (
        "What is the main topic of this "
        "research paper?"
    )

    retrieved_chunks = retrieve_relevant_chunks(
        client=client,
        embedder=embedder,
        reranker=reranker,
        collection_name=collection_name,
        keyword_search=keyword_search,
        query=query,
        dense_top_k=5,
        keyword_top_k=5,
        rrf_top_k=10,
        final_top_k=3,
    )

    assert retrieved_chunks
    assert len(retrieved_chunks) <= 3

    for chunk in retrieved_chunks:
        assert chunk["text"]
        assert chunk["document_name"]
        assert chunk["page_number"]
        assert "rerank_score" in chunk

    # --------------------------------------------------
    # 8. Connect real Gemini
    # --------------------------------------------------

    gemini_service = GeminiService()

    generation_service = GenerationService(
        llm_service=gemini_service
    )

    # --------------------------------------------------
    # 9. Create complete RAG service
    # --------------------------------------------------

    rag_service = RAGService(
        client=client,
        embedder=embedder,
        reranker=reranker,
        collection_name=collection_name,
        keyword_search=keyword_search,
        generation_service=generation_service,
        dense_top_k=5,
        keyword_top_k=5,
        rrf_top_k=10,
        final_top_k=3,
    )

    # --------------------------------------------------
    # 10. Generate final answer
    # --------------------------------------------------

    result = rag_service.answer(
        query=query
    )

    assert result["answer"]
    assert isinstance(
        result["answer"],
        str,
    )

    assert result["sources"]
    assert len(result["sources"]) <= 3

    # --------------------------------------------------
    # 11. Validate returned source metadata
    # --------------------------------------------------

    for source in result["sources"]:
        assert source["text"]
        assert source["document_id"]
        assert source["document_name"]
        assert source["page_number"]
        assert "rerank_score" in source
