from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.base import Base
from app.database.ingestion_repository import (
    IngestionRepository,
)
from app.database.models import (
    IngestionDocumentModel,
    IngestionJobModel,
)


def create_database():
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


def add_job(
    session,
    job_id,
    total_documents,
):
    job = IngestionJobModel(
        job_id=job_id,
        total_documents=total_documents,
        status="queued",
    )

    session.add(job)
    session.commit()

    return job


def add_document(
    session,
    job_id,
    document_id,
    status,
):
    document = IngestionDocumentModel(
        job_id=job_id,
        document_id=document_id,
        document_name=f"{document_id}.pdf",
        status=status,
        attempt_count=0,
    )

    session.add(document)
    session.commit()

    return document


def test_job_progress_becomes_completed():
    session = create_database()

    add_job(
        session=session,
        job_id="job-1",
        total_documents=2,
    )

    add_document(
        session=session,
        job_id="job-1",
        document_id="doc-1",
        status="completed",
    )

    add_document(
        session=session,
        job_id="job-1",
        document_id="doc-2",
        status="completed",
    )

    repository = IngestionRepository(session)

    job = repository.update_job_progress(
        job_id="job-1"
    )

    assert job.completed_documents == 2
    assert job.failed_documents == 0
    assert job.status == "completed"

    session.close()


def test_job_progress_becomes_completed_with_errors():
    session = create_database()

    add_job(
        session=session,
        job_id="job-2",
        total_documents=3,
    )

    add_document(
        session=session,
        job_id="job-2",
        document_id="doc-1",
        status="completed",
    )

    add_document(
        session=session,
        job_id="job-2",
        document_id="doc-2",
        status="completed",
    )

    add_document(
        session=session,
        job_id="job-2",
        document_id="doc-3",
        status="failed",
    )

    repository = IngestionRepository(session)

    job = repository.update_job_progress(
        job_id="job-2"
    )

    assert job.completed_documents == 2
    assert job.failed_documents == 1
    assert job.status == "completed_with_errors"

    session.close()


def test_job_progress_is_processing_when_documents_are_processing():
    session = create_database()

    add_job(
        session=session,
        job_id="job-3",
        total_documents=2,
    )

    add_document(
        session=session,
        job_id="job-3",
        document_id="doc-1",
        status="processing",
    )

    add_document(
        session=session,
        job_id="job-3",
        document_id="doc-2",
        status="queued",
    )

    repository = IngestionRepository(session)

    job = repository.update_job_progress(
        job_id="job-3"
    )

    assert job.completed_documents == 0
    assert job.failed_documents == 0
    assert job.status == "processing"

    session.close()


def test_job_progress_is_queued_when_all_documents_are_queued():
    session = create_database()

    add_job(
        session=session,
        job_id="job-4",
        total_documents=2,
    )

    add_document(
        session=session,
        job_id="job-4",
        document_id="doc-1",
        status="queued",
    )

    add_document(
        session=session,
        job_id="job-4",
        document_id="doc-2",
        status="queued",
    )

    repository = IngestionRepository(session)

    job = repository.update_job_progress(
        job_id="job-4"
    )

    assert job.completed_documents == 0
    assert job.failed_documents == 0
    assert job.status == "queued"

    session.close()


def test_job_progress_is_processing_when_completed_and_queued_remain():
    session = create_database()

    add_job(
        session=session,
        job_id="job-5",
        total_documents=3,
    )

    add_document(
        session=session,
        job_id="job-5",
        document_id="doc-1",
        status="completed",
    )

    add_document(
        session=session,
        job_id="job-5",
        document_id="doc-2",
        status="queued",
    )

    add_document(
        session=session,
        job_id="job-5",
        document_id="doc-3",
        status="queued",
    )

    repository = IngestionRepository(session)

    job = repository.update_job_progress(
        job_id="job-5"
    )

    assert job.completed_documents == 1
    assert job.failed_documents == 0
    assert job.status == "processing"

    session.close()


def test_job_progress_is_processing_when_failed_and_queued_documents_remain():
    session = create_database()

    add_job(
        session=session,
        job_id="job-6",
        total_documents=2,
    )

    add_document(
        session=session,
        job_id="job-6",
        document_id="doc-1",
        status="failed",
    )

    add_document(
        session=session,
        job_id="job-6",
        document_id="doc-2",
        status="queued",
    )

    repository = IngestionRepository(session)

    job = repository.update_job_progress(
        job_id="job-6"
    )

    assert job.completed_documents == 0
    assert job.failed_documents == 1
    assert job.status == "processing"

    session.close()


def test_job_progress_with_no_documents_is_processing():
    session = create_database()

    add_job(
        session=session,
        job_id="job-7",
        total_documents=2,
    )

    repository = IngestionRepository(session)

    job = repository.update_job_progress(
        job_id="job-7"
    )

    assert job.completed_documents == 0
    assert job.failed_documents == 0
    assert job.status == "processing"

    session.close()
