from app.services.ingestion_worker import IngestionWorker
from app.ingestion.document_identity import generate_document_id
from app.database.models import (
    DocumentModel,
    IngestionDocumentModel,
    IngestionJobModel,
)
from app.database.ingestion_repository import IngestionRepository
from sqlalchemy import select
import pymupdf
from pathlib import Path


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


def test_api_created_job_can_be_processed_by_worker(
    tmp_path: Path,
    api_client,
    test_database,
):
    document_path = tmp_path / "paper1.pdf"
    create_test_pdf(document_path)

    document_id = generate_document_id(
        str(document_path)
    )

    with test_database() as db:
        registered_document = DocumentModel(
            document_id=document_id,
            document_name="paper1.pdf",
            file_type=".pdf",
            document_path=str(document_path),
            page_count=1,
            status="registered",
        )

        db.add(registered_document)
        db.commit()

    response = api_client.post(
        "/ingestion/jobs",
        json={
            "document_ids": [document_id],
        },
    )

    assert response.status_code == 201

    job_id = response.json()["job_id"]

    with test_database() as db:
        repository = IngestionRepository(
            session=db
        )

        qdrant_client = FakeQdrantClient()
        embedder = FakeEmbedder()

        worker = IngestionWorker(
            ingestion_job_service=None,
            qdrant_client=qdrant_client,
            embedder=embedder,
            collection_name="test_collection",
            document_directory=str(tmp_path),
        )

        processed_count = worker.process_queued_documents(
            repository=repository,
            job_id=job_id,
        )

        assert processed_count == 1

        job = db.get(
            IngestionJobModel,
            job_id,
        )

        assert job is not None

        statement = (
            select(IngestionDocumentModel)
            .where(
                IngestionDocumentModel.job_id == job_id
            )
        )

        document = db.scalars(statement).one()

        # Diagnostic output: identify the actual processing failure.
        print("JOB STATUS:", job.status)
        print("DOCUMENT STATUS:", document.status)
        print("DOCUMENT ERROR:", document.error_message)
        print("ATTEMPT COUNT:", document.attempt_count)

        assert job.status == "completed"
        assert job.completed_documents == 1
        assert job.failed_documents == 0

        assert document.document_id == document_id
        assert document.status == "completed"
        assert document.error_message is None
        assert document.attempt_count == 1

        assert len(qdrant_client.indexed_chunks) > 0

