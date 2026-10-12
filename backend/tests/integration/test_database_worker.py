
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.base import Base
from app.database.chunk_repository import ChunkRepository
from app.database.ingestion_repository import IngestionRepository
from app.database.models import (
    DocumentModel,
    IngestionDocumentModel,
    IngestionJobModel,
)
from app.services.ingestion_worker import IngestionWorker


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[0.1, 0.2] for _ in texts]


def create_database():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    return SessionLocal()


def create_worker(tmp_path):
    return IngestionWorker(
        ingestion_job_service=None,
        qdrant_client=None,
        embedder=FakeEmbedder(),
        collection_name="test_collection",
        document_directory=str(tmp_path),
    )


def add_document(
    session,
    job_id,
    document_id,
    document_name,
    status="queued",
):
    document = DocumentModel(
        document_id=document_id,
        document_name=document_name,
        file_type=".pdf",
        document_path=str(
            session.info["document_directory"] / document_name
        ),
        page_count=1,
        status="registered",
    )

    ingestion_document = IngestionDocumentModel(
        job_id=job_id,
        document_id=document_id,
        document_name=document_name,
        status=status,
        attempt_count=0,
    )

    session.add(document)
    session.add(ingestion_document)

    return document, ingestion_document


def test_repository_claims_queued_document_for_worker(tmp_path):
    session = create_database()
    session.info["document_directory"] = tmp_path

    job = IngestionJobModel(
        job_id="job-1",
        total_documents=1,
        status="queued",
    )

    session.add(job)

    add_document(
        session=session,
        job_id="job-1",
        document_id="document-1",
        document_name="paper.pdf",
    )

    session.commit()

    repository = IngestionRepository(session)

    claimed_document = repository.claim_next_queued_document(
        job_id="job-1"
    )

    assert claimed_document is not None
    assert claimed_document.document_id == "document-1"
    assert claimed_document.status == "processing"
    assert claimed_document.attempt_count == 1

    repository.commit()
    session.close()


def test_worker_claims_and_completes_document_and_updates_job(
    tmp_path,
    monkeypatch,
):
    session = create_database()
    session.info["document_directory"] = tmp_path

    job = IngestionJobModel(
        job_id="job-2",
        total_documents=1,
        status="queued",
    )

    session.add(job)

    add_document(
        session=session,
        job_id="job-2",
        document_id="document-2",
        document_name="paper.pdf",
    )

    session.commit()

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"fake pdf")

    repository = IngestionRepository(session)
    worker = create_worker(tmp_path)

    monkeypatch.setattr(
        "app.services.ingestion_worker.extract_text_from_pdf",
        lambda path: {
            "document_id": "document-2",
            "document_name": "paper.pdf",
            "file_type": ".pdf",
            "document_path": str(path),
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "text": "Research content.",
                }
            ],
        },
    )

    monkeypatch.setattr(
        "app.services.ingestion_worker.index_document",
        lambda **kwargs: 1,
    )

    processed = worker.process_next_queued_document(
        repository=repository,
        job_id="job-2",
    )

    assert processed is True

    refreshed_document = session.get(
        IngestionDocumentModel,
        {
            "job_id": "job-2",
            "document_id": "document-2",
        },
    )

    refreshed_job = session.get(
        IngestionJobModel,
        "job-2",
    )

    assert refreshed_document is not None
    assert refreshed_document.status == "completed"
    assert refreshed_document.attempt_count == 1
    assert refreshed_document.error_message is None

    assert refreshed_job is not None
    assert refreshed_job.completed_documents == 1
    assert refreshed_job.failed_documents == 0
    assert refreshed_job.status == "completed"

    chunk_repository = ChunkRepository(session=session)
    persisted_chunks = chunk_repository.get_chunks(
        document_id="document-2"
    )

    assert len(persisted_chunks) > 0
    assert persisted_chunks[0]["document_id"] == "document-2"
    assert persisted_chunks[0]["document_name"] == "paper.pdf"
    assert persisted_chunks[0]["page_number"] == 1
    assert "Research content." in persisted_chunks[0]["text"]

    session.close()


def test_worker_marks_document_failed_and_updates_job(
    tmp_path,
    monkeypatch,
):
    session = create_database()
    session.info["document_directory"] = tmp_path

    job = IngestionJobModel(
        job_id="job-3",
        total_documents=1,
        status="queued",
    )

    session.add(job)

    add_document(
        session=session,
        job_id="job-3",
        document_id="document-3",
        document_name="paper.pdf",
    )

    session.commit()

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"fake pdf")

    repository = IngestionRepository(session)
    worker = create_worker(tmp_path)

    monkeypatch.setattr(
        "app.services.ingestion_worker.extract_text_from_pdf",
        lambda path: {
            "document_id": "document-3",
            "document_name": "paper.pdf",
            "file_type": ".pdf",
            "document_path": str(path),
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "text": "Research content.",
                }
            ],
        },
    )

    def failing_index(**kwargs):
        raise RuntimeError("Qdrant indexing failed.")

    monkeypatch.setattr(
        "app.services.ingestion_worker.index_document",
        failing_index,
    )

    processed = worker.process_next_queued_document(
        repository=repository,
        job_id="job-3",
    )

    assert processed is True

    refreshed_document = session.get(
        IngestionDocumentModel,
        {
            "job_id": "job-3",
            "document_id": "document-3",
        },
    )

    refreshed_job = session.get(
        IngestionJobModel,
        "job-3",
    )

    assert refreshed_document is not None
    assert refreshed_document.status == "failed"
    assert refreshed_document.attempt_count == 1
    assert (
        refreshed_document.error_message
        == "Qdrant indexing failed."
    )

    assert refreshed_job is not None
    assert refreshed_job.completed_documents == 0
    assert refreshed_job.failed_documents == 1
    assert (
        refreshed_job.status
        == "completed_with_errors"
    )

    session.close()


def test_worker_processes_multiple_documents_and_updates_job(
    tmp_path,
    monkeypatch,
):
    session = create_database()
    session.info["document_directory"] = tmp_path

    job = IngestionJobModel(
        job_id="job-4",
        total_documents=3,
        status="queued",
    )

    session.add(job)

    documents = [
        ("document-4a", "paper1.pdf"),
        ("document-4b", "paper2.pdf"),
        ("document-4c", "paper3.pdf"),
    ]

    for document_id, document_name in documents:
        add_document(
            session=session,
            job_id="job-4",
            document_id=document_id,
            document_name=document_name,
        )

        (tmp_path / document_name).write_bytes(b"fake pdf")

    session.commit()

    repository = IngestionRepository(session)
    worker = create_worker(tmp_path)

    def fake_extract(path):
        document_name = path.name

        document_id = {
            "paper1.pdf": "document-4a",
            "paper2.pdf": "document-4b",
            "paper3.pdf": "document-4c",
        }[document_name]

        return {
            "document_id": document_id,
            "document_name": document_name,
            "file_type": ".pdf",
            "document_path": str(path),
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
        fake_extract,
    )

    monkeypatch.setattr(
        "app.services.ingestion_worker.index_document",
        lambda **kwargs: 1,
    )

    processed_count = worker.process_queued_documents(
        repository=repository,
        job_id="job-4",
    )

    assert processed_count == 3

    refreshed_job = session.get(
        IngestionJobModel,
        "job-4",
    )

    assert refreshed_job is not None
    assert refreshed_job.completed_documents == 3
    assert refreshed_job.failed_documents == 0
    assert refreshed_job.status == "completed"

    session.close()


def test_stale_processing_document_can_be_recovered(tmp_path):
    session = create_database()
    session.info["document_directory"] = tmp_path

    job = IngestionJobModel(
        job_id="job-5",
        total_documents=1,
        status="processing",
    )

    session.add(job)

    document = DocumentModel(
        document_id="document-5",
        document_name="paper.pdf",
        file_type=".pdf",
        document_path=str(tmp_path / "paper.pdf"),
        page_count=1,
        status="registered",
    )

    stale_time = (
        datetime.now(timezone.utc)
        - timedelta(minutes=60)
    )

    ingestion_document = IngestionDocumentModel(
        job_id="job-5",
        document_id="document-5",
        document_name="paper.pdf",
        status="processing",
        attempt_count=1,
        updated_at=stale_time,
    )

    session.add(document)
    session.add(ingestion_document)
    session.commit()

    repository = IngestionRepository(session)
    worker = create_worker(tmp_path)

    recovered = worker.recover_stale_documents(
        repository=repository,
        timeout_minutes=30,
    )

    assert recovered == 1

    refreshed = session.get(
        IngestionDocumentModel,
        {
            "job_id": "job-5",
            "document_id": "document-5",
        },
    )

    assert refreshed is not None
    assert refreshed.status == "queued"
    assert (
        refreshed.error_message
        == "Recovered after worker timeout."
    )

    session.close()
