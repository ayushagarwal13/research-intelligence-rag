from app.generation.context_builder import build_context


def test_build_context():

    chunks = [
        {
            "chunk_id": "chunk-001",
            "document_name": "rag.pdf",
            "page_number": 4,
            "text": (
                "RAG retrieves external information."
            ),
        },
        {
            "chunk_id": "chunk-002",
            "document_name": "retrieval.pdf",
            "page_number": 8,
            "text": (
                "Retrieval quality affects "
                "RAG performance."
            ),
        },
    ]

    context = build_context(chunks)

    assert "[Source 1]" in context
    assert "[Source 2]" in context

    assert "rag.pdf" in context
    assert "retrieval.pdf" in context

    assert "Page: 4" in context
    assert "Page: 8" in context

    assert (
        "RAG retrieves external information."
        in context
    )


def test_empty_chunks_return_empty_context():

    context = build_context([])

    assert context == ""


def test_chunks_without_text_are_skipped():

    chunks = [
        {
            "chunk_id": "chunk-001",
            "document_name": "rag.pdf",
            "page_number": 4,
            "text": "",
        },
        {
            "chunk_id": "chunk-002",
            "document_name": "rag.pdf",
            "page_number": 5,
            "text": "Useful RAG information.",
        },
    ]

    context = build_context(chunks)

    assert "Useful RAG information." in context
    assert "Page: 4" not in context
