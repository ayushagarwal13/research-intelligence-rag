from unittest.mock import MagicMock

from app.models.ingestion import (
    IngestionDocument,
    IngestionJob,
)
from app.services.ingestion_submission_service import (
    IngestionSubmissionService,
)


def create_service():
    document_repository = MagicMock()
    ingestion_repository = MagicMock()

    service = IngestionSubmissionService(
        document_repository=document_repository,
        ingestion_repository=ingestion_repository,
    )

    return (
        service,
        document_repository,
    )


def test_submit_documents_registers_documents_and_creates_job():
    service, _ = create_service()

    registration_service = MagicMock()
    job_service = MagicMock()

    service.document_registration_service = (
        registration_service
    )

    service.ingestion_job_service = job_service

    document_1 = MagicMock()
    document_1.document_id = "document-001"
    document_1.document_name = "paper1.pdf"

    document_2 = MagicMock()
    document_2.document_id = "document-002"
    document_2.document_name = "paper2.pdf"

    registration_service.register_document.side_effect = [
        document_1,
        document_2,
    ]

    expected_job = IngestionJob(
        job_id="job-001",
        total_documents=2,
    )

    expected_ingestion_documents = [
        IngestionDocument(
            document_id="document-001",
            job_id="job-001",
            document_name="paper1.pdf",
        ),
        IngestionDocument(
            document_id="document-002",
            job_id="job-001",
            document_name="paper2.pdf",
        ),
    ]

    job_service.create_job.return_value = (
        expected_job,
        expected_ingestion_documents,
    )

    job, ingestion_documents = (
        service.submit_documents(
            file_paths=[
                "data/documents/paper1.pdf",
                "data/documents/paper2.pdf",
            ]
        )
    )

    assert job is expected_job
    assert (
        ingestion_documents
        is expected_ingestion_documents
    )

    assert (
        registration_service.register_document.call_count
        == 2
    )

    registration_service.register_document.assert_any_call(
        "data/documents/paper1.pdf"
    )

    registration_service.register_document.assert_any_call(
        "data/documents/paper2.pdf"
    )

    job_service.create_job.assert_called_once_with(
        document_names=[
            "paper1.pdf",
            "paper2.pdf",
        ],
        document_ids=[
            "document-001",
            "document-002",
        ],
    )


def test_submit_documents_rejects_empty_file_list():
    service, _ = create_service()

    try:
        service.submit_documents([])

    except ValueError as error:
        assert (
            str(error)
            == "At least one document is required."
        )

    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_submit_documents_propagates_registration_failure():
    service, _ = create_service()

    registration_service = MagicMock()
    job_service = MagicMock()

    service.document_registration_service = (
        registration_service
    )

    service.ingestion_job_service = job_service

    registration_service.register_document.side_effect = (
        FileNotFoundError(
            "Document not found."
        )
    )

    try:
        service.submit_documents(
            ["data/documents/missing.pdf"]
        )

    except FileNotFoundError as error:
        assert (
            str(error)
            == "Document not found."
        )

    else:
        raise AssertionError(
            "Expected FileNotFoundError"
        )

    job_service.create_job.assert_not_called()


def test_submit_registered_documents_creates_job():
    service, document_repository = (
        create_service()
    )

    job_service = MagicMock()

    service.ingestion_job_service = (
        job_service
    )

    document_1 = MagicMock()
    document_1.document_id = "document-001"
    document_1.document_name = "paper1.pdf"

    document_2 = MagicMock()
    document_2.document_id = "document-002"
    document_2.document_name = "paper2.pdf"

    document_repository.get_document.side_effect = [
        document_1,
        document_2,
    ]

    expected_job = IngestionJob(
        job_id="job-001",
        total_documents=2,
    )

    expected_documents = [
        IngestionDocument(
            document_id="document-001",
            job_id="job-001",
            document_name="paper1.pdf",
        ),
        IngestionDocument(
            document_id="document-002",
            job_id="job-001",
            document_name="paper2.pdf",
        ),
    ]

    job_service.create_job.return_value = (
        expected_job,
        expected_documents,
    )

    job, documents = (
        service.submit_registered_documents(
            [
                "document-001",
                "document-002",
            ]
        )
    )

    assert job is expected_job
    assert documents is expected_documents

    assert (
        document_repository.get_document.call_count
        == 2
    )

    job_service.create_job.assert_called_once_with(
        document_names=[
            "paper1.pdf",
            "paper2.pdf",
        ],
        document_ids=[
            "document-001",
            "document-002",
        ],
    )


def test_submit_registered_documents_rejects_empty_list():
    service, _ = create_service()

    try:
        service.submit_registered_documents([])

    except ValueError as error:
        assert (
            str(error)
            == "At least one document is required."
        )

    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_submit_registered_documents_rejects_duplicate_ids():
    service, _ = create_service()

    try:
        service.submit_registered_documents(
            [
                "document-001",
                "document-001",
            ]
        )

    except ValueError as error:
        assert (
            str(error)
            == (
                "Duplicate document IDs are not "
                "allowed within the same job."
            )
        )

    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_submit_registered_documents_rejects_missing_document():
    service, document_repository = (
        create_service()
    )

    document_repository.get_document.return_value = (
        None
    )

    try:
        service.submit_registered_documents(
            ["document-001"]
        )

    except ValueError as error:
        assert (
            str(error)
            == "Document not found: document-001"
        )

    else:
        raise AssertionError(
            "Expected ValueError"
        )
