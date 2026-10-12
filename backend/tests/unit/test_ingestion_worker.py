import pytest

from app.models.ingestion import (
    IngestionDocument,
    IngestionJob,
)
from app.services.ingestion_worker import (
    IngestionWorker,
)


class FakeIngestionJobService:
    """
    Fake ingestion service used to test the worker
    without requiring a database.
    """

    def __init__(self):
        self.processing_calls = []
        self.completed_calls = []
        self.failed_calls = []

    def mark_document_processing(
        self,
        document,
    ):
        self.processing_calls.append(
            document
        )

        document.status = "processing"
        document.attempt_count += 1

    def mark_document_completed(
        self,
        document,
        job,
        documents,
    ):
        self.completed_calls.append(
            (
                document,
                job,
                documents,
            )
        )

        document.status = "completed"

    def mark_document_failed(
        self,
        document,
        job,
        documents,
        error_message,
    ):
        self.failed_calls.append(
            (
                document,
                job,
                documents,
                error_message,
            )
        )

        document.status = "failed"
        document.error_message = (
            error_message
        )


class FakeEmbedder:
    """
    Fake embedder used to avoid loading the real
    BGE model during worker unit tests.
    """

    def embed_texts(
        self,
        texts,
    ):
        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]


class FakeQdrantClient:
    """
    Fake Qdrant client.

    Qdrant behavior is tested separately.
    """

    pass


def create_worker(
    ingestion_service=None,
    document_directory="data/documents",
):
    if ingestion_service is None:
        ingestion_service = (
            FakeIngestionJobService()
        )

    return IngestionWorker(
        ingestion_job_service=ingestion_service,
        qdrant_client=FakeQdrantClient(),
        embedder=FakeEmbedder(),
        collection_name="research_documents",
        document_directory=document_directory,
    )


def create_job_and_document(
    document_name="paper.pdf",
    document_id="document-123",
):
    job = IngestionJob(
        job_id="job-123",
        total_documents=1,
    )

    document = IngestionDocument(
        document_id=document_id,
        job_id=job.job_id,
        document_name=document_name,
    )

    return job, document


def test_missing_document_is_marked_failed(
    tmp_path,
):
    service = FakeIngestionJobService()

    worker = create_worker(
        ingestion_service=service,
        document_directory=str(tmp_path),
    )

    job, document = create_job_and_document(
        document_name="missing.pdf"
    )

    with pytest.raises(FileNotFoundError):
        worker.process_document(
            document=document,
            job=job,
            documents=[document],
        )

    assert len(
        service.processing_calls
    ) == 1

    assert len(
        service.completed_calls
    ) == 0

    assert len(
        service.failed_calls
    ) == 1

    assert document.status == "failed"

    assert (
        "Document not found"
        in document.error_message
    )


def test_document_identity_mismatch_is_rejected(
    tmp_path,
    monkeypatch,
):
    service = FakeIngestionJobService()

    document_path = (
        tmp_path / "paper.pdf"
    )

    document_path.write_bytes(
        b"fake pdf content"
    )

    worker = create_worker(
        ingestion_service=service,
        document_directory=str(tmp_path),
    )

    job, document = create_job_and_document(
        document_name="paper.pdf",
        document_id="expected-document-id",
    )

    fake_extracted_document = {
        "document_id": "different-document-id",
        "document_name": "paper.pdf",
        "file_type": ".pdf",
        "document_path": str(document_path),
        "page_count": 1,
        "pages": [
            {
                "page_number": 1,
                "text": "Research content.",
            }
        ],
    }

    monkeypatch.setattr(
        "app.services.ingestion_worker.extract_text_from_pdf",
        lambda path: fake_extracted_document,
    )

    with pytest.raises(ValueError):
        worker.process_document(
            document=document,
            job=job,
            documents=[document],
        )

    assert document.status == "failed"

    assert (
        "Document identity mismatch"
        in document.error_message
    )

    assert len(
        service.completed_calls
    ) == 0

    assert len(
        service.failed_calls
    ) == 1


def test_successful_document_is_marked_completed(
    tmp_path,
    monkeypatch,
):
    service = FakeIngestionJobService()

    document_path = (
        tmp_path / "paper.pdf"
    )

    document_path.write_bytes(
        b"fake pdf content"
    )

    worker = create_worker(
        ingestion_service=service,
        document_directory=str(tmp_path),
    )

    job, document = create_job_and_document(
        document_name="paper.pdf",
        document_id="document-123",
    )

    fake_extracted_document = {
        "document_id": "document-123",
        "document_name": "paper.pdf",
        "file_type": ".pdf",
        "document_path": str(document_path),
        "page_count": 1,
        "pages": [
            {
                "page_number": 1,
                "text": "Research content.",
            }
        ],
    }

    monkeypatch.setattr(
        "app.services.ingestion_worker.extract_text_from_pdf",
        lambda path: fake_extracted_document,
    )

    index_calls = []

    def fake_index_document(
        client,
        embedder,
        collection_name,
        document,
        chunk_size,
        overlap,
    ):
        index_calls.append(
            {
                "collection_name": collection_name,
                "document": document,
                "chunk_size": chunk_size,
                "overlap": overlap,
            }
        )

    monkeypatch.setattr(
        "app.services.ingestion_worker.index_document",
        fake_index_document,
    )

    worker.process_document(
        document=document,
        job=job,
        documents=[document],
    )

    assert document.status == "completed"

    assert len(
        service.processing_calls
    ) == 1

    assert len(
        service.completed_calls
    ) == 1

    assert len(
        service.failed_calls
    ) == 0

    assert len(index_calls) == 1

    assert (
        index_calls[0]["collection_name"]
        == "research_documents"
    )

    assert (
        index_calls[0]["document"]["document_id"]
        == "document-123"
    )


def test_indexing_failure_is_marked_failed(
    tmp_path,
    monkeypatch,
):
    service = FakeIngestionJobService()

    document_path = (
        tmp_path / "paper.pdf"
    )

    document_path.write_bytes(
        b"fake pdf content"
    )

    worker = create_worker(
        ingestion_service=service,
        document_directory=str(tmp_path),
    )

    job, document = create_job_and_document(
        document_name="paper.pdf",
        document_id="document-123",
    )

    fake_extracted_document = {
        "document_id": "document-123",
        "document_name": "paper.pdf",
        "file_type": ".pdf",
        "document_path": str(document_path),
        "page_count": 1,
        "pages": [
            {
                "page_number": 1,
                "text": "Research content.",
            }
        ],
    }

    monkeypatch.setattr(
        "app.services.ingestion_worker.extract_text_from_pdf",
        lambda path: fake_extracted_document,
    )

    def failing_index_document(
        *args,
        **kwargs,
    ):
        raise RuntimeError(
            "Qdrant indexing failed."
        )

    monkeypatch.setattr(
        "app.services.ingestion_worker.index_document",
        failing_index_document,
    )

    with pytest.raises(
        RuntimeError,
        match="Qdrant indexing failed",
    ):
        worker.process_document(
            document=document,
            job=job,
            documents=[document],
        )

    assert document.status == "failed"

    assert (
        "Qdrant indexing failed"
        in document.error_message
    )

    assert len(
        service.completed_calls
    ) == 0

    assert len(
        service.failed_calls
    ) == 1


def test_document_attempt_count_increments_when_processing_starts():
    service = FakeIngestionJobService()

    worker = create_worker(
        ingestion_service=service
    )

    job, document = create_job_and_document()

    assert document.attempt_count == 0
    assert document.status == "queued"

    worker.ingestion_job_service.mark_document_processing(
        document
    )

    assert document.attempt_count == 1
    assert document.status == "processing"

    worker.ingestion_job_service.mark_document_processing(
        document
    )

    assert document.attempt_count == 2
    assert document.status == "processing"

    assert len(
        service.processing_calls
    ) == 2
