from pathlib import Path

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
    In-memory repository for testing the ingestion worker.
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


class FakeDocumentProcessor:
    """
    Fake processing dependency used to test
    batch orchestration without loading BGE/Qdrant.
    """

    def __init__(
        self,
        failing_document_name=None,
    ):
        self.failing_document_name = (
            failing_document_name
        )
        self.processed_documents = []

    def process_document(
        self,
        document,
        job,
        documents,
    ):
        self.processed_documents.append(
            document.document_name
        )

        if (
            document.document_name
            == self.failing_document_name
        ):
            raise RuntimeError(
                "Simulated processing failure."
            )


def create_document_file(
    file_path: Path,
    content: str,
) -> str:
    """
    Create a small file and return its SHA-256 ID.
    """

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    return generate_document_id(
        str(file_path)
    )


def test_process_job_processes_all_queued_documents(
    tmp_path,
):
    repository = FakeRepository()

    ingestion_service = IngestionJobService(
        repository
    )

    document_paths = [
        tmp_path / "paper-1.pdf",
        tmp_path / "paper-2.pdf",
        tmp_path / "paper-3.pdf",
    ]

    document_ids = [
        create_document_file(
            document_paths[0],
            "paper one",
        ),
        create_document_file(
            document_paths[1],
            "paper two",
        ),
        create_document_file(
            document_paths[2],
            "paper three",
        ),
    ]

    job, documents = (
        ingestion_service.create_job(
            document_names=[
                path.name
                for path in document_paths
            ],
            document_ids=document_ids,
        )
    )

    worker = IngestionWorker(
        ingestion_job_service=ingestion_service,
        qdrant_client=None,
        embedder=None,
        collection_name="test",
        document_directory=str(tmp_path),
    )

    processed = []

    def fake_process_document(
        document,
        job,
        documents,
    ):
        processed.append(
            document.document_name
        )

        ingestion_service.mark_document_processing(
            document
        )

        ingestion_service.mark_document_completed(
            document=document,
            job=job,
            documents=documents,
        )

    worker.process_document = (
        fake_process_document
    )

    worker.process_job(
        job=job,
        documents=documents,
    )

    assert processed == [
        "paper-1.pdf",
        "paper-2.pdf",
        "paper-3.pdf",
    ]

    assert job.status == "completed"

    assert job.completed_documents == 3

    assert job.failed_documents == 0


def test_process_job_continues_after_document_failure(
    tmp_path,
):
    repository = FakeRepository()

    ingestion_service = IngestionJobService(
        repository
    )

    document_paths = [
        tmp_path / "paper-1.pdf",
        tmp_path / "paper-2.pdf",
        tmp_path / "paper-3.pdf",
    ]

    document_ids = [
        create_document_file(
            document_paths[0],
            "paper one",
        ),
        create_document_file(
            document_paths[1],
            "paper two",
        ),
        create_document_file(
            document_paths[2],
            "paper three",
        ),
    ]

    job, documents = (
        ingestion_service.create_job(
            document_names=[
                path.name
                for path in document_paths
            ],
            document_ids=document_ids,
        )
    )

    worker = IngestionWorker(
        ingestion_job_service=ingestion_service,
        qdrant_client=None,
        embedder=None,
        collection_name="test",
        document_directory=str(tmp_path),
    )

    processed = []

    def fake_process_document(
        document,
        job,
        documents,
    ):
        processed.append(
            document.document_name
        )

        ingestion_service.mark_document_processing(
            document
        )

        if (
            document.document_name
            == "paper-2.pdf"
        ):
            ingestion_service.mark_document_failed(
                document=document,
                job=job,
                documents=documents,
                error_message="Simulated failure.",
            )

            raise RuntimeError(
                "Simulated failure."
            )

        ingestion_service.mark_document_completed(
            document=document,
            job=job,
            documents=documents,
        )

    worker.process_document = (
        fake_process_document
    )

    worker.process_job(
        job=job,
        documents=documents,
    )

    assert processed == [
        "paper-1.pdf",
        "paper-2.pdf",
        "paper-3.pdf",
    ]

    assert documents[0].status == "completed"

    assert documents[1].status == "failed"

    assert documents[2].status == "completed"

    assert (
        documents[1].error_message
        == "Simulated failure."
    )

    assert job.status == "completed_with_errors"

    assert job.completed_documents == 2

    assert job.failed_documents == 1


def test_process_job_skips_documents_not_in_queued_state(
    tmp_path,
):
    repository = FakeRepository()

    ingestion_service = IngestionJobService(
        repository
    )

    document_paths = [
        tmp_path / "paper-1.pdf",
        tmp_path / "paper-2.pdf",
    ]

    document_ids = [
        create_document_file(
            document_paths[0],
            "paper one",
        ),
        create_document_file(
            document_paths[1],
            "paper two",
        ),
    ]

    job, documents = (
        ingestion_service.create_job(
            document_names=[
                path.name
                for path in document_paths
            ],
            document_ids=document_ids,
        )
    )

    documents[0].status = "completed"

    worker = IngestionWorker(
        ingestion_job_service=ingestion_service,
        qdrant_client=None,
        embedder=None,
        collection_name="test",
        document_directory=str(tmp_path),
    )

    processed = []

    def fake_process_document(
        document,
        job,
        documents,
    ):
        processed.append(
            document.document_name
        )

        ingestion_service.mark_document_processing(
            document
        )

        ingestion_service.mark_document_completed(
            document=document,
            job=job,
            documents=documents,
        )

    worker.process_document = (
        fake_process_document
    )

    worker.process_job(
        job=job,
        documents=documents,
    )

    assert processed == [
        "paper-2.pdf"
    ]

    assert documents[0].status == "completed"

    assert documents[1].status == "completed"
