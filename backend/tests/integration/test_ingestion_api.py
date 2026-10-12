from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.dependencies import get_db
from app.database.base import Base
from app.database.models import (
    DocumentModel,
    IngestionDocumentModel,
    IngestionJobModel,
)
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


def override_get_db() -> Generator[
    Session,
    None,
    None,
]:
    db = TestSessionLocal()

    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = (
    override_get_db
)

client = TestClient(app)


def setup_test_database() -> None:
    Base.metadata.create_all(
        bind=test_engine
    )


def teardown_test_database() -> None:
    Base.metadata.drop_all(
        bind=test_engine
    )


def create_registered_document(
    db: Session,
    document_id: str,
    document_name: str,
) -> None:
    document = DocumentModel(
        document_id=document_id,
        document_name=document_name,
        file_type=".pdf",
        document_path=(
            f"data/documents/{document_name}"
        ),
        page_count=10,
        status="registered",
    )

    db.add(document)
    db.commit()


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


def test_create_ingestion_job():
    setup_test_database()

    try:
        with TestSessionLocal() as db:
            create_registered_document(
                db=db,
                document_id="document-001",
                document_name="paper1.pdf",
            )

            create_registered_document(
                db=db,
                document_id="document-002",
                document_name="paper2.pdf",
            )

        response = client.post(
            "/ingestion/jobs",
            json={
                "document_ids": [
                    "document-001",
                    "document-002",
                ]
            },
        )

        assert response.status_code == 201

        response_data = response.json()

        assert (
            response_data["status"]
            == "queued"
        )

        assert (
            response_data["total_documents"]
            == 2
        )

        job_id = response_data["job_id"]

        with TestSessionLocal() as db:
            job = db.get(
                IngestionJobModel,
                job_id,
            )

            assert job is not None

            assert (
                job.total_documents
                == 2
            )

            assert (
                job.status
                == "queued"
            )

            statement = select(
                IngestionDocumentModel
            ).where(
                IngestionDocumentModel.job_id
                == job_id
            )

            ingestion_documents = list(
                db.scalars(
                    statement
                ).all()
            )

            assert len(
                ingestion_documents
            ) == 2

            document_ids = {
                document.document_id
                for document
                in ingestion_documents
            }

            assert document_ids == {
                "document-001",
                "document-002",
            }

            assert all(
                document.status == "queued"
                for document
                in ingestion_documents
            )

    finally:
        teardown_test_database()


def test_create_ingestion_job_rejects_duplicate_document_ids():
    setup_test_database()

    try:
        with TestSessionLocal() as db:
            create_registered_document(
                db=db,
                document_id="document-001",
                document_name="paper1.pdf",
            )

        response = client.post(
            "/ingestion/jobs",
            json={
                "document_ids": [
                    "document-001",
                    "document-001",
                ]
            },
        )

        assert response.status_code == 400

        assert response.json() == {
            "detail": (
                "Duplicate document IDs are not "
                "allowed within the same job."
            )
        }

    finally:
        teardown_test_database()


def test_create_ingestion_job_rejects_missing_document():
    setup_test_database()

    try:
        response = client.post(
            "/ingestion/jobs",
            json={
                "document_ids": [
                    "does-not-exist"
                ]
            },
        )

        assert response.status_code == 400

        assert response.json() == {
            "detail": (
                "Document not found: "
                "does-not-exist"
            )
        }

    finally:
        teardown_test_database()
