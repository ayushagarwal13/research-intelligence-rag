from unittest.mock import Mock

from app.services.ingestion_worker import IngestionWorker


def create_worker(chunk_repository):
    """
    Create an ingestion worker with mocked external
    dependencies.
    """

    return IngestionWorker(
        ingestion_job_service=Mock(),
        qdrant_client=Mock(),
        embedder=Mock(),
        collection_name="research_documents",
        document_directory="data/documents",
        chunk_size=500,
        overlap=50,
        chunk_repository=chunk_repository,
    )


def test_persist_chunks_saves_chunks_to_repository():
    """
    Verify that the worker generates chunks from an
    extracted document and passes them to the
    ChunkRepository.
    """

    chunk_repository = Mock()

    worker = create_worker(
        chunk_repository=chunk_repository
    )

    extracted_document = {
        "document_id": "a" * 64,
        "document_name": "research-paper.pdf",
        "file_type": ".pdf",
        "document_path": (
            "data/documents/research-paper.pdf"
        ),
        "page_count": 1,
        "pages": [
            {
                "page_number": 1,
                "text": (
                    "Retrieval augmented generation "
                    "combines retrieval with generation. "
                    "Hybrid retrieval combines dense "
                    "and keyword search."
                ),
            }
        ],
    }

    worker._persist_chunks(
        extracted_document=extracted_document
    )

    chunk_repository.create_chunks.assert_called_once()

    persisted_chunks = (
        chunk_repository
        .create_chunks
        .call_args.args[0]
    )

    assert len(persisted_chunks) > 0

    assert all(
        chunk["document_id"]
        == extracted_document["document_id"]
        for chunk in persisted_chunks
    )

    assert all(
        chunk["document_name"]
        == extracted_document["document_name"]
        for chunk in persisted_chunks
    )


def test_persist_chunks_does_nothing_without_repository():
    """
    Verify backward compatibility when no chunk
    repository is configured.
    """

    worker = IngestionWorker(
        ingestion_job_service=Mock(),
        qdrant_client=Mock(),
        embedder=Mock(),
        collection_name="research_documents",
        document_directory="data/documents",
        chunk_size=500,
        overlap=50,
        chunk_repository=None,
    )

    extracted_document = {
        "document_id": "a" * 64,
        "document_name": "research-paper.pdf",
        "file_type": ".pdf",
        "document_path": (
            "data/documents/research-paper.pdf"
        ),
        "page_count": 1,
        "pages": [
            {
                "page_number": 1,
                "text": (
                    "Retrieval augmented generation "
                    "combines retrieval with generation."
                ),
            }
        ],
    }

    # This should complete without raising an exception.
    worker._persist_chunks(
        extracted_document=extracted_document
    )
