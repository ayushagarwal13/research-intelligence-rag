from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database.ingestion_repository import (
    IngestionRepository,
)
from app.database.models import (
    IngestionDocumentModel,
    IngestionJobModel,
)


def create_test_session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={
            "check_same_thread": False
        },
        poolclass=StaticPool,
    )

    Base.metadata.create_all(
        bind=engine
    )

    return Session(engine)


def test_get_queued_jobs_returns_jobs_with_queued_documents():
    with create_test_session() as session:
        first_job = IngestionJobModel(
            job_id="job-001",
            total_documents=2,
            status="queued",
            created_at=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            updated_at=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
        )

        second_job = IngestionJobModel(
            job_id="job-002",
            total_documents=2,
            status="processing",
            created_at=datetime(
                2026,
                1,
                2,
                tzinfo=timezone.utc,
            ),
            updated_at=datetime(
                2026,
                1,
                2,
                tzinfo=timezone.utc,
            ),
        )

        completed_job = IngestionJobModel(
            job_id="job-003",
            total_documents=1,
            status="completed",
            created_at=datetime(
                2026,
                1,
                3,
                tzinfo=timezone.utc,
            ),
            updated_at=datetime(
                2026,
                1,
                3,
                tzinfo=timezone.utc,
            ),
        )

        session.add_all(
            [
                first_job,
                second_job,
                completed_job,
            ]
        )

        session.add_all(
            [
                IngestionDocumentModel(
                    job_id="job-001",
                    document_id="document-001",
                    document_name="paper1.pdf",
                    status="queued",
                    attempt_count=0,
                ),
                IngestionDocumentModel(
                    job_id="job-001",
                    document_id="document-002",
                    document_name="paper2.pdf",
                    status="queued",
                    attempt_count=0,
                ),
                IngestionDocumentModel(
                    job_id="job-002",
                    document_id="document-003",
                    document_name="paper3.pdf",
                    status="queued",
                    attempt_count=0,
                ),
                IngestionDocumentModel(
                    job_id="job-003",
                    document_id="document-004",
                    document_name="paper4.pdf",
                    status="completed",
                    attempt_count=1,
                ),
            ]
        )

        session.commit()

        repository = IngestionRepository(
            session=session
        )

        jobs = repository.get_queued_jobs()

        job_ids = [
            job.job_id
            for job in jobs
        ]

        assert job_ids == [
            "job-001",
            "job-002",
        ]


def test_get_queued_jobs_does_not_return_completed_job():
    with create_test_session() as session:
        job = IngestionJobModel(
            job_id="job-001",
            total_documents=1,
            status="completed",
        )

        document = IngestionDocumentModel(
            job_id="job-001",
            document_id="document-001",
            document_name="paper.pdf",
            status="completed",
            attempt_count=1,
        )

        session.add(job)
        session.add(document)
        session.commit()

        repository = IngestionRepository(
            session=session
        )

        jobs = repository.get_queued_jobs()

        assert jobs == []
