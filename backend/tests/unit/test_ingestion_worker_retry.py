from pathlib import Path

import pytest

from app.ingestion.document_identity import (
    generate_document_id,
)
from app.services.ingestion_job_service import (
    IngestionJobService,
)
from app.services.ingestion_worker import (
    IngestionWorker,
)


class FakeRepository:
    """
    In-memory repository for testing retry behavior.
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


def create_job_with_documents(
    tmp_path,
    number_of_documents=1,
):
    repository = FakeRepository()

    ingestion_service = IngestionJobService(
        repository
    )

    document_paths = []

    for index in range(number_of_documents):
        document_path = (
            tmp_path
            / f"paper-{index + 1}.pdf"
        )

        document_paths.append(
            document_path
        )

    document_ids = [
        create_document_file(
            path,
            f"paper {index + 1}",
        )
        for index, path in enumerate(
            document_paths
        )
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

    return (
        repository,
        ingestion_service,
        job,
        documents,
    )


def create_worker(
    tmp_path,
    ingestion_service,
):
    return IngestionWorker(
        ingestion_job_service=ingestion_service,
        qdrant_client=None,
        embedder=None,
        collection_name="test",
        document_directory=str(tmp_path),
    )


def test_retry_document_reprocesses_failed_document(
    tmp_path,
):
    (
        repository,
        ingestion_service,
        job,
        documents,
    ) = create_job_with_documents(
        tmp_path
    )

    document = documents[0]

    ingestion_service.mark_document_processing(
        document
    )

    ingestion_service.mark_document_failed(
        document=document,
        job=job,
        documents=documents,
        error_message="Initial failure.",
    )

    assert document.status == "failed"
    assert document.attempt_count == 1

    worker = create_worker(
        tmp_path,
        ingestion_service,
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

    worker.retry_document(
        document=document,
        job=job,
        documents=documents,
    )

    assert processed == [
        "paper-1.pdf"
    ]

    assert document.status == "completed"

    assert document.attempt_count == 2

    assert document.error_message is None

    assert job.status == "completed"

    assert job.completed_documents == 1

    assert job.failed_documents == 0


def test_retry_document_failure_returns_to_failed(
    tmp_path,
):
    (
        repository,
        ingestion_service,
        job,
        documents,
    ) = create_job_with_documents(
        tmp_path
    )

    document = documents[0]

    ingestion_service.mark_document_processing(
        document
    )

    ingestion_service.mark_document_failed(
        document=document,
        job=job,
        documents=documents,
        error_message="Initial failure.",
    )

    worker = create_worker(
        tmp_path,
        ingestion_service,
    )

    def failing_process_document(
        document,
        job,
        documents,
    ):
        ingestion_service.mark_document_processing(
            document
        )

        ingestion_service.mark_document_failed(
            document=document,
            job=job,
            documents=documents,
            error_message="Retry failure.",
        )

        raise RuntimeError(
            "Retry failure."
        )

    worker.process_document = (
        failing_process_document
    )

    with pytest.raises(
        RuntimeError,
        match="Retry failure.",
    ):
        worker.retry_document(
            document=document,
            job=job,
            documents=documents,
        )

    assert document.status == "failed"

    assert document.attempt_count == 2

    assert (
        document.error_message
        == "Retry failure."
    )

    assert job.status == "completed_with_errors"

    assert job.completed_documents == 0

    assert job.failed_documents == 1


def test_retry_document_respects_maximum_attempts(
    tmp_path,
):
    (
        repository,
        ingestion_service,
        job,
        documents,
    ) = create_job_with_documents(
        tmp_path
    )

    document = documents[0]

    document.attempt_count = 3
    document.status = "failed"
    document.error_message = (
        "Maximum attempts reached."
    )

    worker = create_worker(
        tmp_path,
        ingestion_service,
    )

    with pytest.raises(
        ValueError,
        match="Maximum ingestion attempts exceeded.",
    ):
        worker.retry_document(
            document=document,
            job=job,
            documents=documents,
        )

    assert document.status == "failed"

    assert document.attempt_count == 3


def test_retry_failed_documents_retries_only_failed_documents(
    tmp_path,
):
    (
        repository,
        ingestion_service,
        job,
        documents,
    ) = create_job_with_documents(
        tmp_path,
        number_of_documents=3,
    )

    completed_document = documents[0]

    failed_document = documents[1]

    exhausted_document = documents[2]

    completed_document.status = "completed"
    completed_document.attempt_count = 1

    failed_document.status = "failed"
    failed_document.attempt_count = 1
    failed_document.error_message = (
        "Temporary failure."
    )

    exhausted_document.status = "failed"
    exhausted_document.attempt_count = 3
    exhausted_document.error_message = (
        "Permanent failure."
    )

    job.completed_documents = 1
    job.failed_documents = 2
    job.status = "completed_with_errors"

    worker = create_worker(
        tmp_path,
        ingestion_service,
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

    worker.retry_failed_documents(
        job=job,
        documents=documents,
    )

    assert processed == [
        "paper-2.pdf"
    ]

    assert (
        completed_document.status
        == "completed"
    )

    assert (
        failed_document.status
        == "completed"
    )

    assert (
        exhausted_document.status
        == "failed"
    )

    assert (
        failed_document.attempt_count
        == 2
    )

    assert (
        exhausted_document.attempt_count
        == 3
    )

    # One document remains permanently failed,
    # so the job correctly remains completed_with_errors.
    assert (
        job.status
        == "completed_with_errors"
    )

    assert job.completed_documents == 2

    assert job.failed_documents == 1
