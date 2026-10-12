import pytest
from unittest.mock import Mock
from app.database.ingestion_repository import (
    IngestionRepository,
)
from app.models.ingestion import (
    MAX_INGESTION_ATTEMPTS,
)
from app.services.ingestion_job_service import (
    IngestionJobService,
)


class FakeRepository:
    """
    In-memory repository used to test service logic
    without a real database.
    """

    def __init__(self):
        self.jobs = {}
        self.documents = {}
        self.commit_count = 0

    def create_job(self, job):
        self.jobs[job.job_id] = job

    def create_documents(self, documents):
        for document in documents:
            self.documents[
                document.document_id
            ] = document

    def update_job(self, job):
        if job.job_id not in self.jobs:
            raise ValueError(
                f"Job not found: {job.job_id}"
            )

        self.jobs[job.job_id] = job

    def update_document(self, document):
        if document.document_id not in self.documents:
            raise ValueError(
                f"Document not found: "
                f"{document.document_id}"
            )

        self.documents[
            document.document_id
        ] = document

    def commit(self):
        self.commit_count += 1


def create_service():

    repository = FakeRepository()

    service = IngestionJobService(
        repository=repository
    )

    return service, repository


def create_job_and_documents():

    service, repository = create_service()

    job, documents = service.create_job(
        ["paper.pdf"]
    )

    return (
        service,
        repository,
        job,
        documents,
    )


def test_create_job():

    service, repository = create_service()

    job, documents = service.create_job(
        [
            "paper1.pdf",
            "paper2.pdf",
            "paper3.pdf",
        ]
    )

    assert job.total_documents == 3
    assert job.completed_documents == 0
    assert job.failed_documents == 0
    assert job.status == "queued"
    assert job.job_id

    assert len(documents) == 3

    assert job.job_id in repository.jobs

    assert all(
        document.document_id
        in repository.documents
        for document in documents
    )


def test_create_job_requires_documents():

    service, repository = create_service()

    with pytest.raises(ValueError):

        service.create_job([])


def test_create_documents():

    service, repository = create_service()

    job, documents = service.create_job(
        [
            "paper1.pdf",
            "paper2.pdf",
        ]
    )

    assert len(documents) == 2

    assert documents[0].job_id == job.job_id
    assert documents[1].job_id == job.job_id

    assert documents[0].status == "queued"
    assert documents[0].attempt_count == 0


def test_document_can_start_processing():

    (
        service,
        repository,
        job,
        documents,
    ) = create_job_and_documents()

    document = documents[0]

    service.mark_document_processing(
        document
    )

    assert document.status == "processing"
    assert document.attempt_count == 1

    stored_document = repository.documents[
        document.document_id
    ]

    assert stored_document.status == "processing"
    assert stored_document.attempt_count == 1

    assert repository.commit_count == 2


def test_processing_document_can_complete():

    (
        service,
        repository,
        job,
        documents,
    ) = create_job_and_documents()

    document = documents[0]

    service.mark_document_processing(
        document
    )

    service.mark_document_completed(
        document,
        job,
        documents,
    )

    assert document.status == "completed"
    assert document.attempt_count == 1

    assert job.completed_documents == 1
    assert job.failed_documents == 0
    assert job.status == "completed"

    assert repository.documents[
        document.document_id
    ].status == "completed"

    assert repository.jobs[
        job.job_id
    ].status == "completed"


def test_processing_document_can_fail():

    (
        service,
        repository,
        job,
        documents,
    ) = create_job_and_documents()

    document = documents[0]

    service.mark_document_processing(
        document
    )

    service.mark_document_failed(
        document,
        job,
        documents,
        "PDF could not be extracted.",
    )

    assert document.status == "failed"
    assert document.attempt_count == 1

    assert (
        document.error_message
        == "PDF could not be extracted."
    )

    assert job.completed_documents == 0
    assert job.failed_documents == 1
    assert job.status == "completed_with_errors"

    assert repository.documents[
        document.document_id
    ].status == "failed"

    assert repository.jobs[
        job.job_id
    ].status == "completed_with_errors"


def test_failed_document_can_be_retried():

    (
        service,
        repository,
        job,
        documents,
    ) = create_job_and_documents()

    document = documents[0]

    service.mark_document_processing(
        document
    )

    service.mark_document_failed(
        document,
        job,
        documents,
        "Temporary processing error.",
    )

    service.retry_document(
        document
    )

    assert document.status == "queued"
    assert document.error_message is None
    assert document.attempt_count == 1

    assert repository.documents[
        document.document_id
    ].status == "queued"


def test_retry_increments_attempt_when_processing_restarts():

    (
        service,
        repository,
        job,
        documents,
    ) = create_job_and_documents()

    document = documents[0]

    service.mark_document_processing(
        document
    )

    service.mark_document_failed(
        document,
        job,
        documents,
        "Temporary error.",
    )

    service.retry_document(
        document
    )

    service.mark_document_processing(
        document
    )

    assert document.status == "processing"
    assert document.attempt_count == 2


def test_document_can_be_processed_three_times():

    (
        service,
        repository,
        job,
        documents,
    ) = create_job_and_documents()

    document = documents[0]

    for attempt in range(
        1,
        MAX_INGESTION_ATTEMPTS + 1,
    ):

        service.mark_document_processing(
            document
        )

        assert document.attempt_count == attempt

        service.mark_document_failed(
            document,
            job,
            documents,
            f"Failure on attempt {attempt}.",
        )

        if attempt < MAX_INGESTION_ATTEMPTS:

            service.retry_document(
                document
            )

            assert document.status == "queued"

        else:

            assert document.status == "failed"


def test_document_cannot_exceed_max_attempts():

    (
        service,
        repository,
        job,
        documents,
    ) = create_job_and_documents()

    document = documents[0]

    for _ in range(
        MAX_INGESTION_ATTEMPTS
    ):

        service.mark_document_processing(
            document
        )

        service.mark_document_failed(
            document,
            job,
            documents,
            "Permanent failure.",
        )

        if (
            document.attempt_count
            < MAX_INGESTION_ATTEMPTS
        ):

            service.retry_document(
                document
            )

    assert (
        document.attempt_count
        == MAX_INGESTION_ATTEMPTS
    )

    assert document.status == "failed"

    with pytest.raises(ValueError):

        service.retry_document(
            document
        )


def test_queued_document_cannot_be_completed():

    (
        service,
        repository,
        job,
        documents,
    ) = create_job_and_documents()

    document = documents[0]

    with pytest.raises(ValueError):

        service.mark_document_completed(
            document,
            job,
            documents,
        )


def test_queued_document_cannot_fail():

    (
        service,
        repository,
        job,
        documents,
    ) = create_job_and_documents()

    document = documents[0]

    with pytest.raises(ValueError):

        service.mark_document_failed(
            document,
            job,
            documents,
            "Error",
        )


def test_completed_document_cannot_be_retried():

    (
        service,
        repository,
        job,
        documents,
    ) = create_job_and_documents()

    document = documents[0]

    service.mark_document_processing(
        document
    )

    service.mark_document_completed(
        document,
        job,
        documents,
    )

    with pytest.raises(ValueError):

        service.retry_document(
            document
        )


def test_processing_document_cannot_start_again():

    (
        service,
        repository,
        job,
        documents,
    ) = create_job_and_documents()

    document = documents[0]

    service.mark_document_processing(
        document
    )

    with pytest.raises(ValueError):

        service.mark_document_processing(
            document
        )


def test_job_stays_processing_until_all_documents_finish():

    service, repository = create_service()

    job, documents = service.create_job(
        [
            "paper1.pdf",
            "paper2.pdf",
            "paper3.pdf",
        ]
    )

    service.mark_document_processing(
        documents[0]
    )

    service.mark_document_completed(
        documents[0],
        job,
        documents,
    )

    assert job.completed_documents == 1
    assert job.status == "processing"

    service.mark_document_processing(
        documents[1]
    )

    service.mark_document_completed(
        documents[1],
        job,
        documents,
    )

    assert job.completed_documents == 2
    assert job.status == "processing"

    service.mark_document_processing(
        documents[2]
    )

    service.mark_document_completed(
        documents[2],
        job,
        documents,
    )

    assert job.completed_documents == 3
    assert job.failed_documents == 0
    assert job.status == "completed"


def test_job_can_complete_with_errors():

    service, repository = create_service()

    job, documents = service.create_job(
        [
            "paper1.pdf",
            "paper2.pdf",
        ]
    )

    service.mark_document_processing(
        documents[0]
    )

    service.mark_document_completed(
        documents[0],
        job,
        documents,
    )

    service.mark_document_processing(
        documents[1]
    )

    service.mark_document_failed(
        documents[1],
        job,
        documents,
        "Invalid PDF.",
    )

    assert job.completed_documents == 1
    assert job.failed_documents == 1
    assert job.status == "completed_with_errors"


def test_successful_retry_removes_previous_failure():

    service, repository = create_service()

    job, documents = service.create_job(
        ["paper.pdf"]
    )

    document = documents[0]

    service.mark_document_processing(
        document
    )

    service.mark_document_failed(
        document,
        job,
        documents,
        "Temporary error.",
    )

    assert job.failed_documents == 1
    assert job.completed_documents == 0
    assert job.status == "completed_with_errors"

    service.retry_document(
        document
    )

    service.mark_document_processing(
        document
    )

    service.mark_document_completed(
        document,
        job,
        documents,
    )

    assert job.completed_documents == 1
    assert job.failed_documents == 0
    assert job.status == "completed"



def test_create_job_rejects_empty_document_ids():
    service, _ = create_service()

    with pytest.raises(
        ValueError,
        match="At least one document ID is required.",
    ):
        service.create_job(
            ["paper.pdf"],
            document_ids=[],
        )


def test_create_job_rejects_mismatched_document_ids():
    service, _ = create_service()

    with pytest.raises(
        ValueError,
        match="Number of document names must match",
    ):
        service.create_job(
            ["paper1.pdf", "paper2.pdf"],
            document_ids=["doc-1"],
        )


def test_create_job_rejects_duplicate_document_ids():
    service, _ = create_service()

    with pytest.raises(
        ValueError,
        match="Duplicate document IDs",
    ):
        service.create_job(
            ["paper1.pdf", "paper2.pdf"],
            document_ids=["doc-1", "doc-1"],
        )


def test_create_job_rejects_blank_document_id():
    service, _ = create_service()

    with pytest.raises(
        ValueError,
        match="Document IDs cannot be empty.",
    ):
        service.create_job(
            ["paper.pdf"],
            document_ids=["   "],
        )


def test_mark_document_failed_rejects_blank_error():
    service, _, job, documents = create_job_and_documents()
    document = documents[0]

    service.mark_document_processing(document)

    with pytest.raises(
        ValueError,
        match="Error message cannot be empty.",
    ):
        service.mark_document_failed(
            document,
            job,
            documents,
            "   ",
        )


def test_retry_with_job_recalculates_progress():
    service, repository, job, documents = create_job_and_documents()
    document = documents[0]

    service.mark_document_processing(document)
    service.mark_document_failed(
        document,
        job,
        documents,
        "Temporary error.",
    )

    service.retry_document(
        document,
        job=job,
        documents=documents,
    )

    assert document.status == "queued"
    assert job.failed_documents == 0
    assert job.completed_documents == 0
    assert job.status == "queued"
    assert repository.jobs[job.job_id].status == "queued"


def test_job_progress_is_processing_when_document_is_processing():
    service, _, job, documents = create_job_and_documents()
    document = documents[0]

    service.mark_document_processing(document)

    service._update_job_progress(job, documents)

    assert job.status == "processing"
    assert job.completed_documents == 0
    assert job.failed_documents == 0


def test_job_progress_uses_fallback_for_unrecognized_document_states():
    service, _, job, _ = create_job_and_documents()

    # An empty document list does not match any of the normal
    # progress conditions for a job expecting one document.
    service._update_job_progress(job, [])

    assert job.status == "processing"
    assert job.completed_documents == 0
    assert job.failed_documents == 0




def test_create_job_uses_preferred_repository_method():
    repository = Mock()
    repository.create_ingestion_documents = Mock()

    service = IngestionJobService(repository)

    job, documents = service.create_job(
        ["paper.pdf"],
        document_ids=["doc-1"],
    )

    repository.create_job.assert_called_once_with(job)
    repository.create_ingestion_documents.assert_called_once_with(
        documents
    )
    repository.commit.assert_called_once()


def test_create_job_raises_when_repository_has_no_document_method():
    repository = Mock()
    repository.create_ingestion_documents = None
    repository.create_documents = None

    service = IngestionJobService(repository)

    with pytest.raises(
        AttributeError,
        match="Repository must provide either",
    ):
        service.create_job(
            ["paper.pdf"],
            document_ids=["doc-1"],
        )


def test_legacy_document_ids_are_deterministic():
    service, _ = create_service()

    _, first_documents = service.create_job(["paper.pdf"])
    _, second_documents = service.create_job(["paper.pdf"])

    assert (
        first_documents[0].document_id
        == second_documents[0].document_id
    )


def test_document_cannot_start_processing_after_max_attempts():
    service, _, _, documents = create_job_and_documents()
    document = documents[0]

    document.attempt_count = MAX_INGESTION_ATTEMPTS

    with pytest.raises(
        ValueError,
        match="Maximum ingestion attempts exceeded.",
    ):
        service.mark_document_processing(document)

    assert document.status == "queued"
    assert document.attempt_count == MAX_INGESTION_ATTEMPTS
