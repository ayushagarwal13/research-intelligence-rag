from app.services.worker_runner import WorkerRunner
from app.services.ingestion_worker import IngestionWorker
from app.services.ingestion_job_service import IngestionJobService
from app.ingestion.document_identity import generate_document_id
from app.database.models import (
    IngestionDocumentModel,
    IngestionJobModel,
)
from app.database.ingestion_repository import IngestionRepository
from app.database.base import Base
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import pymupdf
from pathlib import Path


TEST_DATABASE_URL = "sqlite://"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestSessionLocal = sessionmaker(
    bind=test_engine,
    autoflush=False,
    autocommit=False,
)


class FakeEmbedder:
    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [1.0] * 768
            for _ in texts
        ]


class FakeQdrantClient:
    def __init__(self):
        self.indexed_chunks = []

    def upsert(
        self,
        collection_name,
        points,
        **kwargs,
    ):
        self.indexed_chunks.extend(points)


def create_test_pdf(
    file_path: Path,
) -> None:
    """Create a minimal valid PDF for integration testing."""
    document = pymupdf.open()

    page = document.new_page()
    page.insert_text(
        (72, 72),
        "This is a research document used for testing.",
    )

    document.save(file_path)
    document.close()


def setup_test_database() -> None:
    Base.metadata.create_all(
        bind=test_engine
    )


def teardown_test_database() -> None:
    Base.metadata.drop_all(
        bind=test_engine
    )


def test_worker_runner_processes_queued_job(
    tmp_path: Path,
    monkeypatch,
):
    setup_test_database()

    try:
        document_path = tmp_path / "paper1.pdf"
        create_test_pdf(document_path)

        document_id = generate_document_id(
            str(document_path)
        )

        with TestSessionLocal() as db:
            repository = IngestionRepository(
                session=db
            )

            job_service = IngestionJobService(
                repository=repository
            )

            job, _ = job_service.create_job(
                document_names=["paper1.pdf"],
                document_ids=[document_id],
            )

        fake_qdrant_client = FakeQdrantClient()

        worker = IngestionWorker(
            ingestion_job_service=None,
            qdrant_client=fake_qdrant_client,
            embedder=FakeEmbedder(),
            collection_name="test_collection",
            document_directory=str(tmp_path),
        )

        monkeypatch.setattr(
            "app.services.worker_runner.SessionLocal",
            TestSessionLocal,
        )

        runner = WorkerRunner(
            worker=worker,
            poll_interval_seconds=1,
        )

        processed_count = runner.process_once()

        assert processed_count == 1

        with TestSessionLocal() as db:
            job = db.get(
                IngestionJobModel,
                job.job_id,
            )

            assert job is not None

            document = db.get(
                IngestionDocumentModel,
                {
                    "job_id": job.job_id,
                    "document_id": document_id,
                },
            )

            assert document is not None

            # Diagnostic output for any processing failure.
            print("JOB STATUS:", job.status)
            print("DOCUMENT STATUS:", document.status)
            print("DOCUMENT ERROR:", document.error_message)
            print("ATTEMPT COUNT:", document.attempt_count)

            assert job.status == "completed"
            assert job.completed_documents == 1
            assert job.failed_documents == 0

            assert document.status == "completed"
            assert document.error_message is None
            assert document.attempt_count == 1

        assert len(fake_qdrant_client.indexed_chunks) > 0

    finally:
        teardown_test_database()

