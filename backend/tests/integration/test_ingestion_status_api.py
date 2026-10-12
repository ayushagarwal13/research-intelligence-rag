from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.dependencies import get_db
from app.database.base import Base
from app.database.models import IngestionJobModel
from app.main import app


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


def override_get_db():
    with TestSessionLocal() as session:
        yield session


def setup_test_database():
    Base.metadata.create_all(
        bind=test_engine
    )


def teardown_test_database():
    Base.metadata.drop_all(
        bind=test_engine
    )


def test_get_ingestion_job_status():
    setup_test_database()

    try:
        with TestSessionLocal() as db:
            job = IngestionJobModel(
                job_id="job-status-001",
                total_documents=10,
                completed_documents=7,
                failed_documents=1,
                status="processing",
            )

            db.add(job)
            db.commit()

        app.dependency_overrides[
            get_db
        ] = override_get_db

        client = TestClient(app)

        response = client.get(
            "/ingestion/jobs/job-status-001"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["job_id"] == "job-status-001"
        assert data["status"] == "processing"
        assert data["total_documents"] == 10
        assert data["completed_documents"] == 7
        assert data["failed_documents"] == 1

    finally:
        app.dependency_overrides.clear()
        teardown_test_database()


def test_get_ingestion_job_status_returns_404():
    setup_test_database()

    try:
        app.dependency_overrides[
            get_db
        ] = override_get_db

        client = TestClient(app)

        response = client.get(
            "/ingestion/jobs/does-not-exist"
        )

        assert response.status_code == 404

        data = response.json()

        assert (
            data["detail"]
            == "Ingestion job not found: does-not-exist"
        )

    finally:
        app.dependency_overrides.clear()
        teardown_test_database()
