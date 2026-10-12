from datetime import datetime, timedelta, timezone

import pytest
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


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:"
    )

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


def create_job(
    session,
    job_id="job-1",
):
    job = IngestionJobModel(
        job_id=job_id,
        total_documents=1,
        completed_documents=0,
        failed_documents=0,
        status="processing",
    )

    session.add(job)
    session.commit()

    return job


def create_document(
    session,
    job_id="job-1",
    document_id="document-1",
    status="processing",
    attempt_count=1,
    updated_at=None,
    error_message=None,
):
    document = IngestionDocumentModel(
        job_id=job_id,
        document_id=document_id,
        document_name=f"{document_id}.pdf",
        status=status,
        attempt_count=attempt_count,
        updated_at=(
            updated_at
            or datetime.now(timezone.utc)
        ),
        error_message=error_message,
    )

    session.add(document)
    session.commit()

    return document


def test_recover_stale_processing_document(
    session,
):
    create_job(session)

    stale_time = (
        datetime.now(timezone.utc)
        - timedelta(minutes=60)
    )

    create_document(
        session,
        updated_at=stale_time,
        attempt_count=1,
    )

    repository = IngestionRepository(
        session
    )

    recovered_count = (
        repository.recover_stale_processing_documents(
            timeout_minutes=30
        )
    )

    repository.commit()

    assert recovered_count == 1

    stored_document = session.get(
        IngestionDocumentModel,
        (
            "job-1",
            "document-1",
        ),
    )

    assert stored_document.status == "queued"

    assert (
        stored_document.attempt_count
        == 1
    )

    assert (
        stored_document.error_message
        == "Recovered after worker timeout."
    )


def test_recent_processing_document_is_not_recovered(
    session,
):
    create_job(session)

    recent_time = (
        datetime.now(timezone.utc)
        - timedelta(minutes=5)
    )

    create_document(
        session,
        updated_at=recent_time,
    )

    repository = IngestionRepository(
        session
    )

    recovered_count = (
        repository.recover_stale_processing_documents(
            timeout_minutes=30
        )
    )

    repository.commit()

    assert recovered_count == 0

    stored_document = session.get(
        IngestionDocumentModel,
        (
            "job-1",
            "document-1",
        ),
    )

    assert (
        stored_document.status
        == "processing"
    )


def test_completed_document_is_not_recovered(
    session,
):
    create_job(session)

    stale_time = (
        datetime.now(timezone.utc)
        - timedelta(minutes=60)
    )

    create_document(
        session,
        status="completed",
        updated_at=stale_time,
    )

    repository = IngestionRepository(
        session
    )

    recovered_count = (
        repository.recover_stale_processing_documents(
            timeout_minutes=30
        )
    )

    repository.commit()

    assert recovered_count == 0

    stored_document = session.get(
        IngestionDocumentModel,
        (
            "job-1",
            "document-1",
        ),
    )

    assert (
        stored_document.status
        == "completed"
    )


def test_failed_document_is_not_recovered(
    session,
):
    create_job(session)

    stale_time = (
        datetime.now(timezone.utc)
        - timedelta(minutes=60)
    )

    create_document(
        session,
        status="failed",
        updated_at=stale_time,
        error_message="Original failure.",
    )

    repository = IngestionRepository(
        session
    )

    recovered_count = (
        repository.recover_stale_processing_documents(
            timeout_minutes=30
        )
    )

    repository.commit()

    assert recovered_count == 0

    stored_document = session.get(
        IngestionDocumentModel,
        (
            "job-1",
            "document-1",
        ),
    )

    assert (
        stored_document.status
        == "failed"
    )

    assert (
        stored_document.error_message
        == "Original failure."
    )


def test_recovery_preserves_attempt_count(
    session,
):
    create_job(session)

    stale_time = (
        datetime.now(timezone.utc)
        - timedelta(minutes=60)
    )

    create_document(
        session,
        attempt_count=2,
        updated_at=stale_time,
    )

    repository = IngestionRepository(
        session
    )

    repository.recover_stale_processing_documents(
        timeout_minutes=30
    )

    repository.commit()

    stored_document = session.get(
        IngestionDocumentModel,
        (
            "job-1",
            "document-1",
        ),
    )

    assert (
        stored_document.status
        == "queued"
    )

    assert (
        stored_document.attempt_count
        == 2
    )


def test_recovery_updates_timestamp(
    session,
):
    create_job(session)

    old_timestamp = (
        datetime.now(timezone.utc)
        - timedelta(minutes=60)
    )

    create_document(
        session,
        updated_at=old_timestamp,
    )

    repository = IngestionRepository(
        session
    )

    repository.recover_stale_processing_documents(
        timeout_minutes=30
    )

    repository.commit()

    stored_document = session.get(
        IngestionDocumentModel,
        (
            "job-1",
            "document-1",
        ),
    )

    assert stored_document.updated_at is not None

    # SQLite returns DateTime values without timezone
    # information even when timezone=True is configured.
    # Compare normalized UTC-naive timestamps so the test
    # works consistently with the local SQLite test database.
    stored_timestamp = (
        stored_document.updated_at
    )

    if stored_timestamp.tzinfo is not None:
        stored_timestamp = (
            stored_timestamp.astimezone(
                timezone.utc
            ).replace(tzinfo=None)
        )

    expected_timestamp = (
        old_timestamp.astimezone(
            timezone.utc
        ).replace(tzinfo=None)
    )

    assert (
        stored_timestamp
        > expected_timestamp
    )


def test_invalid_timeout_is_rejected(
    session,
):
    repository = IngestionRepository(
        session
    )

    with pytest.raises(
        ValueError,
        match="timeout_minutes must be greater than zero.",
    ):
        repository.recover_stale_processing_documents(
            timeout_minutes=0
        )


def test_multiple_stale_documents_are_recovered(
    session,
):
    create_job(
        session,
        job_id="job-1",
    )

    stale_time = (
        datetime.now(timezone.utc)
        - timedelta(minutes=60)
    )

    create_document(
        session,
        job_id="job-1",
        document_id="document-1",
        updated_at=stale_time,
    )

    create_document(
        session,
        job_id="job-1",
        document_id="document-2",
        updated_at=stale_time,
    )

    repository = IngestionRepository(
        session
    )

    recovered_count = (
        repository.recover_stale_processing_documents(
            timeout_minutes=30
        )
    )

    repository.commit()

    assert recovered_count == 2

    documents = session.query(
        IngestionDocumentModel
    ).all()

    assert len(documents) == 2

    assert all(
        document.status == "queued"
        for document in documents
    )
