from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.base import Base
from app.database.ingestion_repository import (
    IngestionRepository,
)
from app.database.models import (
    DocumentModel,
)
from app.models.ingestion import (
    IngestionDocument,
    IngestionJob,
)
from app.services.ingestion_job_service import (
    IngestionJobService,
)


def create_test_session():
    """
    Create a fresh in-memory SQLite database
    for the integration test.
    """

    engine = create_engine(
        "sqlite:///:memory:"
    )

    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    return SessionLocal()


def create_canonical_document(
    document_id: str,
    document_name: str,
) -> DocumentModel:
    """
    Create a canonical document record.
    """

    return DocumentModel(
        document_id=document_id,
        document_name=document_name,
        file_type=".pdf",
        document_path=(
            f"data/documents/{document_name}"
        ),
        page_count=10,
        status="registered",
    )


def test_complete_ingestion_lifecycle_is_persisted():
    session = create_test_session()

    repository = IngestionRepository(
        session
    )

    service = IngestionJobService(
        repository
    )

    document_id = "a" * 64

    canonical_document = create_canonical_document(
        document_id=document_id,
        document_name="paper.pdf",
    )

    repository.create_document(
        canonical_document
    )
    repository.commit()

    job, documents = service.create_job(
        document_names=["paper.pdf"],
        document_ids=[document_id],
    )

    assert job.status == "queued"
    assert job.total_documents == 1

    assert len(documents) == 1

    document = documents[0]

    assert document.document_id == document_id
    assert document.status == "queued"
    assert document.attempt_count == 0

    service.mark_document_processing(
        document
    )

    stored_document = (
        repository.get_ingestion_document(
            document_id=document_id,
            job_id=job.job_id,
        )
    )

    assert stored_document is not None
    assert stored_document.status == "processing"
    assert stored_document.attempt_count == 1

    service.mark_document_completed(
        document=document,
        job=job,
        documents=documents,
    )

    stored_document = (
        repository.get_ingestion_document(
            document_id=document_id,
            job_id=job.job_id,
        )
    )

    stored_job = repository.get_job(
        job.job_id
    )

    assert stored_document is not None
    assert stored_document.status == "completed"

    assert stored_job is not None
    assert stored_job.status == "completed"
    assert stored_job.completed_documents == 1
    assert stored_job.failed_documents == 0

    session.close()


def test_failed_document_and_retry_are_persisted():
    session = create_test_session()

    repository = IngestionRepository(
        session
    )

    service = IngestionJobService(
        repository
    )

    document_id = "b" * 64

    canonical_document = create_canonical_document(
        document_id=document_id,
        document_name="failed-paper.pdf",
    )

    repository.create_document(
        canonical_document
    )
    repository.commit()

    job, documents = service.create_job(
        document_names=["failed-paper.pdf"],
        document_ids=[document_id],
    )

    document = documents[0]

    service.mark_document_processing(
        document
    )

    service.mark_document_failed(
        document=document,
        job=job,
        documents=documents,
        error_message="Embedding service failed.",
    )

    stored_document = (
        repository.get_ingestion_document(
            document_id=document_id,
            job_id=job.job_id,
        )
    )

    assert stored_document is not None
    assert stored_document.status == "failed"
    assert (
        stored_document.error_message
        == "Embedding service failed."
    )
    assert stored_document.attempt_count == 1

    stored_job = repository.get_job(
        job.job_id
    )

    assert stored_job is not None
    assert (
        stored_job.status
        == "completed_with_errors"
    )
    assert stored_job.failed_documents == 1

    service.retry_document(
        document
    )

    stored_document = (
        repository.get_ingestion_document(
            document_id=document_id,
            job_id=job.job_id,
        )
    )

    assert stored_document is not None
    assert stored_document.status == "queued"
    assert stored_document.error_message is None
    assert stored_document.attempt_count == 1

    session.close()


def test_same_canonical_document_can_be_ingested_again():
    session = create_test_session()

    repository = IngestionRepository(
        session
    )

    service = IngestionJobService(
        repository
    )

    document_id = "c" * 64

    canonical_document = create_canonical_document(
        document_id=document_id,
        document_name="research-paper.pdf",
    )

    repository.create_document(
        canonical_document
    )
    repository.commit()

    first_job, first_documents = (
        service.create_job(
            document_names=[
                "research-paper.pdf"
            ],
            document_ids=[document_id],
        )
    )

    second_job, second_documents = (
        service.create_job(
            document_names=[
                "research-paper.pdf"
            ],
            document_ids=[document_id],
        )
    )

    assert first_job.job_id != second_job.job_id

    assert (
        first_documents[0].document_id
        == document_id
    )

    assert (
        second_documents[0].document_id
        == document_id
    )

    first_record = (
        repository.get_ingestion_document(
            document_id=document_id,
            job_id=first_job.job_id,
        )
    )

    second_record = (
        repository.get_ingestion_document(
            document_id=document_id,
            job_id=second_job.job_id,
        )
    )

    assert first_record is not None
    assert second_record is not None

    assert (
        first_record.document_id
        == second_record.document_id
    )

    assert (
        first_record.job_id
        != second_record.job_id
    )

    session.close()
