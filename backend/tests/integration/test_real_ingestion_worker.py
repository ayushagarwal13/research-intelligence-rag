from pathlib import Path

import pymupdf
from qdrant_client import QdrantClient

from app.database.qdrant_client import (
    create_collection,
)
from app.embeddings.embedder import BGEEmbedder
from app.ingestion.document_identity import (
    generate_document_id,
)
from app.models.ingestion import (
    IngestionDocument,
    IngestionJob,
)
from app.services.ingestion_job_service import (
    IngestionJobService,
)
from app.services.ingestion_worker import (
    IngestionWorker,
)


class FakeRepository:
    """
    In-memory repository used to test the worker
    without PostgreSQL/SQLite.
    """

    def __init__(self):
        self.jobs = {}
        self.documents = {}

    def create_job(self, job):
        self.jobs[job.job_id] = job

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def update_job(self, job):
        self.jobs[job.job_id] = job

    def create_ingestion_documents(
        self,
        documents,
    ):
        for document in documents:
            key = (
                document.job_id,
                document.document_id,
            )

            self.documents[key] = document

    def create_ingestion_document(
        self,
        document,
    ):
        key = (
            document.job_id,
            document.document_id,
        )

        self.documents[key] = document

    def get_ingestion_document(
        self,
        document_id,
        job_id,
    ):
        return self.documents.get(
            (
                job_id,
                document_id,
            )
        )

    def update_document(
        self,
        document,
    ):
        key = (
            document.job_id,
            document.document_id,
        )

        self.documents[key] = document

    def commit(self):
        pass


def create_test_pdf(
    file_path: Path,
) -> None:
    """
    Create a small real PDF for the integration test.
    """

    document = pymupdf.open()

    page = document.new_page()

    page.insert_text(
        (72, 72),
        "Retrieval augmented generation combines "
        "retrieval with language model generation. "
        "The retrieval step provides external context "
        "that can improve factual grounding.",
    )

    document.save(file_path)
    document.close()


def test_real_pdf_is_processed_and_indexed(
    tmp_path,
):
    """
    Verify the real ingestion path:

        PDF
        -> extraction
        -> cleaning
        -> chunking
        -> BGE embeddings
        -> Qdrant
    """

    pdf_path = (
        tmp_path / "research-paper.pdf"
    )

    create_test_pdf(pdf_path)

    document_id = generate_document_id(
        str(pdf_path)
    )

    repository = FakeRepository()

    ingestion_service = IngestionJobService(
        repository
    )

    job, documents = (
        ingestion_service.create_job(
            document_names=[
                pdf_path.name
            ],
            document_ids=[
                document_id
            ],
        )
    )

    client = QdrantClient(":memory:")

    collection_name = (
        "research_documents_test"
    )

    create_collection(
        client=client,
        collection_name=collection_name,
    )

    embedder = BGEEmbedder()

    worker = IngestionWorker(
        ingestion_job_service=(
            ingestion_service
        ),
        qdrant_client=client,
        embedder=embedder,
        collection_name=collection_name,
        document_directory=str(tmp_path),
    )

    worker.process_document(
        document=documents[0],
        job=job,
        documents=documents,
    )

    stored_document = (
        repository.get_ingestion_document(
            document_id=document_id,
            job_id=job.job_id,
        )
    )

    assert stored_document is not None

    assert (
        stored_document.status
        == "completed"
    )

    assert (
        stored_document.attempt_count
        == 1
    )

    collection_info = (
        client.get_collection(
            collection_name
        )
    )

    assert (
        collection_info.points_count
        > 0
    )

    search_result = client.query_points(
        collection_name=collection_name,
        query=embedder.embed_text(
            "How does retrieval improve "
            "language model grounding?"
        ),
        limit=3,
        with_payload=True,
    )

    assert len(
        search_result.points
    ) > 0

    top_result = (
        search_result.points[0]
    )

    assert (
        top_result.payload is not None
    )

    assert (
        top_result.payload["document_id"]
        == document_id
    )

    assert (
        top_result.payload["document_name"]
        == "research-paper.pdf"
    )

    assert (
        top_result.payload["page_number"]
        == 1
    )

    assert (
        top_result.payload["text"]
    )

    assert job.status == "completed"

    assert (
        job.completed_documents
        == 1
    )

    assert (
        job.failed_documents
        == 0
    )
