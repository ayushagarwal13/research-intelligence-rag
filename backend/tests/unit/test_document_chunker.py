from app.chunking.document_chunker import chunk_document


def create_test_document():
    return {
        "document_id": "doc-123",
        "document_name": "research.pdf",
        "file_type": ".pdf",
        "document_path": "data/documents/research.pdf",
        "page_count": 2,
        "pages": [
            {
                "page_number": 1,
                "text": (
                    "This is the first page. "
                    "It contains research information."
                ),
            },
            {
                "page_number": 2,
                "text": (
                    "This is the second page. "
                    "It contains additional information."
                ),
            },
        ],
    }


def test_chunk_document_returns_chunks():
    document = create_test_document()

    chunks = chunk_document(
        document=document,
        chunk_size=100,
        overlap=20,
    )

    assert chunks
    assert isinstance(chunks, list)


def test_chunks_preserve_document_metadata():
    document = create_test_document()

    chunks = chunk_document(
        document=document,
        chunk_size=100,
        overlap=20,
    )

    for chunk in chunks:
        assert chunk["document_id"] == "doc-123"
        assert chunk["document_name"] == "research.pdf"
        assert chunk["page_number"] in [1, 2]
        assert chunk["chunk_id"]
        assert chunk["text"]


def test_chunks_are_generated_for_all_pages():
    document = create_test_document()

    chunks = chunk_document(
        document=document,
        chunk_size=100,
        overlap=20,
    )

    page_numbers = {
        chunk["page_number"]
        for chunk in chunks
    }

    assert 1 in page_numbers
    assert 2 in page_numbers


def test_empty_pages_return_no_chunks():
    document = create_test_document()
    document["pages"] = [
        {
            "page_number": 1,
            "text": "",
        }
    ]

    chunks = chunk_document(document=document)

    assert chunks == []
