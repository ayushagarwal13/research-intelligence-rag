from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database.ingestion_repository import (
    IngestionRepository,
)
from app.database.models import IngestionJobModel


TEST_DATABASE_URL = "sqlite://"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={
        "check_same_thread": False,
    },
    poolclass=StaticPool,
)

TestSessionLocal = sessionmaker(
    bind=test_engine,
    autoflush=False,
    autocommit=False,
)


def setup_test_database() -> None:
    Base.metadata.create_all(
        bind=test_engine
    )


def teardown_test_database() -> None:
    Base.metadata.drop_all(
        bind=test_engine
    )


def test_get_job_model_returns_existing_job():
    setup_test_database()

    try:
        with TestSessionLocal() as db:
            job = IngestionJobModel(
                job_id="job-001",
                total_documents=5,
                completed_documents=2,
                failed_documents=1,
                status="processing",
            )

            db.add(job)
            db.commit()

            repository = IngestionRepository(
                session=db
            )

            result = repository.get_job_model(
                "job-001"
            )

            assert result is not None
            assert result.job_id == "job-001"
            assert result.total_documents == 5
            assert result.completed_documents == 2
            assert result.failed_documents == 1
            assert result.status == "processing"

    finally:
        teardown_test_database()


def test_get_job_model_returns_none_for_missing_job():
    setup_test_database()

    try:
        with TestSessionLocal() as db:
            repository = IngestionRepository(
                session=db
            )

            result = repository.get_job_model(
                "does-not-exist"
            )

            assert result is None

    finally:
        teardown_test_database()
