from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.base import Base
from app.database.ingestion_repository import IngestionRepository
from app.database.models import DocumentModel, IngestionDocumentModel
from app.models.ingestion import IngestionDocument, IngestionJob


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    session = SessionLocal()
    yield session

    session.close()
    engine.dispose()


def create_job(repository, job_id="job-1", total_documents=1):
    job = IngestionJob(
        job_id=job_id,
        total_documents=total_documents,
    )
    repository.create_job(job)
    repository.commit()
    return job


def create_ingestion_document(
    repository,
    job_id="job-1",
    document_id="document-1",
    document_name="paper.pdf",
    status="queued",
    attempt_count=0,
):
    document = IngestionDocument(
        document_id=document_id,
        job_id=job_id,
        document_name=document_name,
        status=status,
        attempt_count=attempt_count,
    )
    repository.create_ingestion_document(document)
    repository.commit()
    return document


def test_create_and_get_job(session):
    repository = IngestionRepository(session)
    job = create_job(repository)

    stored_job = repository.get_job(job.job_id)

    assert stored_job is not None
    assert stored_job.job_id == job.job_id
    assert stored_job.total_documents == 1
    assert stored_job.status == "queued"


def test_create_and_get_ingestion_document(session):
    repository = IngestionRepository(session)
    create_job(repository)
    document = create_ingestion_document(repository)

    stored_document = repository.get_ingestion_document(
        document_id=document.document_id,
        job_id=document.job_id,
    )

    assert stored_document is not None
    assert stored_document.document_id == document.document_id
    assert stored_document.status == "queued"


def test_update_job(session):
    repository = IngestionRepository(session)
    job = create_job(repository)

    job.completed_documents = 1
    job.status = "completed"

    repository.update_job(job)
    repository.commit()

    stored_job = repository.get_job(job.job_id)

    assert stored_job.completed_documents == 1
    assert stored_job.status == "completed"


def test_update_ingestion_document(session):
    repository = IngestionRepository(session)
    create_job(repository)
    document = create_ingestion_document(repository)

    document.status = "completed"
    document.attempt_count = 1

    repository.update_document(document)
    repository.commit()

    stored_document = repository.get_ingestion_document(
        document_id=document.document_id,
        job_id=document.job_id,
    )

    assert stored_document.status == "completed"
    assert stored_document.attempt_count == 1


def test_claim_next_queued_document(session):
    repository = IngestionRepository(session)
    create_job(repository)
    create_ingestion_document(repository)

    claimed_document = repository.claim_next_queued_document(
        job_id="job-1"
    )

    assert claimed_document is not None
    assert claimed_document.document_id == "document-1"
    assert claimed_document.status == "processing"
    assert claimed_document.attempt_count == 1
    assert claimed_document.error_message is None

    repository.commit()

    stored_document = repository.get_ingestion_document(
        document_id="document-1",
        job_id="job-1",
    )

    assert stored_document.status == "processing"
    assert stored_document.attempt_count == 1


def test_claim_returns_documents_in_creation_order(session):
    repository = IngestionRepository(session)
    create_job(repository, total_documents=2)

    create_ingestion_document(
        repository,
        document_id="document-1",
        document_name="paper-1.pdf",
    )
    create_ingestion_document(
        repository,
        document_id="document-2",
        document_name="paper-2.pdf",
    )

    first = repository.claim_next_queued_document(job_id="job-1")
    repository.commit()
    second = repository.claim_next_queued_document(job_id="job-1")
    repository.commit()

    assert first is not None
    assert second is not None
    assert first.document_id == "document-1"
    assert second.document_id == "document-2"


def test_claim_ignores_documents_at_max_attempts(session):
    repository = IngestionRepository(session)
    create_job(repository)

    create_ingestion_document(
        repository,
        document_id="exhausted",
        document_name="exhausted.pdf",
        attempt_count=3,
    )
    create_ingestion_document(
        repository,
        document_id="available",
        document_name="available.pdf",
        attempt_count=1,
    )

    claimed_document = repository.claim_next_queued_document(
        job_id="job-1"
    )

    assert claimed_document is not None
    assert claimed_document.document_id == "available"
    assert claimed_document.attempt_count == 2


def test_claim_returns_none_when_no_queued_document_exists(session):
    repository = IngestionRepository(session)
    create_job(repository)
    create_ingestion_document(
        repository,
        document_id="completed",
        document_name="completed.pdf",
        status="completed",
    )

    claimed_document = repository.claim_next_queued_document(
        job_id="job-1"
    )

    assert claimed_document is None


def test_claim_clears_previous_error_message(session):
    repository = IngestionRepository(session)
    create_job(repository)

    document = IngestionDocument(
        document_id="document-1",
        job_id="job-1",
        document_name="paper.pdf",
        status="queued",
        error_message="Old error",
        attempt_count=1,
    )
    repository.create_ingestion_document(document)
    repository.commit()

    claimed_document = repository.claim_next_queued_document(
        job_id="job-1"
    )

    assert claimed_document is not None
    assert claimed_document.error_message is None
    assert claimed_document.status == "processing"
    assert claimed_document.attempt_count == 2


def test_claim_does_not_claim_processing_document(session):
    repository = IngestionRepository(session)
    create_job(repository)
    create_ingestion_document(
        repository,
        document_id="processing",
        document_name="processing.pdf",
        status="processing",
        attempt_count=1,
    )

    claimed_document = repository.claim_next_queued_document(
        job_id="job-1"
    )

    assert claimed_document is None


def test_update_job_raises_when_job_does_not_exist(session):
    repository = IngestionRepository(session)
    missing_job = IngestionJob(
        job_id="missing-job",
        total_documents=1,
    )

    with pytest.raises(
        ValueError,
        match="Ingestion job not found: missing-job",
    ):
        repository.update_job(missing_job)


def test_create_ingestion_documents_creates_multiple_records(session):
    repository = IngestionRepository(session)
    create_job(repository, total_documents=2)

    documents = [
        IngestionDocument(
            document_id="document-1",
            job_id="job-1",
            document_name="paper-1.pdf",
        ),
        IngestionDocument(
            document_id="document-2",
            job_id="job-1",
            document_name="paper-2.pdf",
        ),
    ]

    repository.create_ingestion_documents(documents)
    repository.commit()

    first = repository.get_ingestion_document(
        job_id="job-1",
        document_id="document-1",
    )
    second = repository.get_ingestion_document(
        job_id="job-1",
        document_id="document-2",
    )

    assert first is not None
    assert second is not None
    assert first.document_name == "paper-1.pdf"
    assert second.document_name == "paper-2.pdf"


def test_create_ingestion_documents_accepts_empty_list(session):
    repository = IngestionRepository(session)

    repository.create_ingestion_documents([])
    repository.commit()

    assert repository.get_documents_for_job("job-1") == []


def test_get_ingestion_document_returns_none_when_missing(session):
    repository = IngestionRepository(session)

    result = repository.get_ingestion_document(
        job_id="job-1",
        document_id="missing-document",
    )

    assert result is None


def test_update_ingestion_document_raises_when_missing(session):
    repository = IngestionRepository(session)
    missing_document = IngestionDocument(
        document_id="missing-document",
        job_id="job-1",
        document_name="missing.pdf",
        status="failed",
        error_message="Processing failed",
        attempt_count=1,
    )

    with pytest.raises(
        ValueError,
        match="Ingestion document not found",
    ):
        repository.update_ingestion_document(missing_document)


def test_update_ingestion_document_updates_all_fields(session):
    repository = IngestionRepository(session)
    create_job(repository)
    create_ingestion_document(repository)

    updated_document = IngestionDocument(
        document_id="document-1",
        job_id="job-1",
        document_name="renamed-paper.pdf",
        status="failed",
        error_message="OCR failed",
        attempt_count=2,
    )

    repository.update_ingestion_document(updated_document)
    repository.commit()

    stored = repository.get_ingestion_document(
        job_id="job-1",
        document_id="document-1",
    )

    assert stored is not None
    assert stored.document_name == "renamed-paper.pdf"
    assert stored.status == "failed"
    assert stored.error_message == "OCR failed"
    assert stored.attempt_count == 2


def test_update_job_progress_raises_when_job_is_missing(session):
    repository = IngestionRepository(session)

    with pytest.raises(
        ValueError,
        match="Ingestion job not found: missing-job",
    ):
        repository.update_job_progress("missing-job")


def test_update_document_model_refreshes_timestamp(session):
    repository = IngestionRepository(session)
    create_job(repository)
    create_ingestion_document(repository)

    model = session.get(
        IngestionDocumentModel,
        {
            "job_id": "job-1",
            "document_id": "document-1",
        },
    )
    assert model is not None

    repository.update_document_model(model)

    assert model.updated_at is not None
    assert model in session


def test_update_job_model_refreshes_timestamp(session):
    repository = IngestionRepository(session)
    create_job(repository)

    model = repository.get_job_model("job-1")
    assert model is not None

    repository.update_job_model(model)

    assert model.updated_at is not None
    assert model in session


def test_create_and_get_document_record(session):
    repository = IngestionRepository(session)

    document = DocumentModel(
        document_id="a" * 64,
        document_name="research-paper.pdf",
        file_type="pdf",
        document_path="data/research-paper.pdf",
        page_count=12,
        status="registered",
    )

    repository.create_document(document)
    repository.commit()

    stored = repository.get_document_record("a" * 64)

    assert stored is not None
    assert stored.document_id == "a" * 64
    assert stored.document_name == "research-paper.pdf"
    assert stored.file_type == "pdf"
    assert stored.document_path == "data/research-paper.pdf"
    assert stored.page_count == 12


def test_create_document_record_and_get_missing_document(session):
    repository = IngestionRepository(session)

    document = DocumentModel(
        document_id="b" * 64,
        document_name="another-paper.pdf",
        file_type="pdf",
        document_path="data/another-paper.pdf",
        page_count=8,
    )

    repository.create_document_record(document)
    repository.commit()

    stored = repository.get_document_record("b" * 64)
    missing = repository.get_document_record("missing")

    assert stored is not None
    assert stored.document_name == "another-paper.pdf"
    assert missing is None


def test_get_job_returns_none_when_missing(session):
    repository = IngestionRepository(session)

    result = repository.get_job("nonexistent-job")

    assert result is None
