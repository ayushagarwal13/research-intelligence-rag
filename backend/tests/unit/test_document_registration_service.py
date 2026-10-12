from pathlib import Path
from unittest.mock import MagicMock, patch

from app.database.models import DocumentModel
from app.services.document_registration_service import (
    DocumentRegistrationService,
)


def test_register_document_creates_document_record(
    tmp_path: Path,
):
    repository = MagicMock()

    repository.get_document.return_value = None

    extracted_document = {
        "document_id": "document-001",
        "document_name": "paper.pdf",
        "file_type": ".pdf",
        "document_path": str(
            tmp_path / "paper.pdf"
        ),
        "page_count": 12,
    }

    document_path = tmp_path / "paper.pdf"
    document_path.write_text(
        "test document"
    )

    service = DocumentRegistrationService(
        repository=repository
    )

    with patch(
        "app.services.document_registration_service.extract_text_from_pdf",
        return_value=extracted_document,
    ):
        document = service.register_document(
            str(document_path)
        )

    assert isinstance(
        document,
        DocumentModel,
    )

    assert (
        document.document_id
        == "document-001"
    )

    assert (
        document.document_name
        == "paper.pdf"
    )

    assert (
        document.page_count
        == 12
    )

    assert (
        document.status
        == "registered"
    )

    repository.create_document.assert_called_once_with(
        document
    )

    repository.commit.assert_called_once_with()


def test_register_document_returns_existing_document(
    tmp_path: Path,
):
    repository = MagicMock()

    existing_document = DocumentModel(
        document_id="document-001",
        document_name="paper.pdf",
        file_type=".pdf",
        document_path=str(
            tmp_path / "paper.pdf"
        ),
        page_count=12,
        status="registered",
    )

    repository.get_document.return_value = (
        existing_document
    )

    extracted_document = {
        "document_id": "document-001",
        "document_name": "paper.pdf",
        "file_type": ".pdf",
        "document_path": str(
            tmp_path / "paper.pdf"
        ),
        "page_count": 12,
    }

    document_path = tmp_path / "paper.pdf"
    document_path.write_text(
        "test document"
    )

    service = DocumentRegistrationService(
        repository=repository
    )

    with patch(
        "app.services.document_registration_service.extract_text_from_pdf",
        return_value=extracted_document,
    ):
        document = service.register_document(
            str(document_path)
        )

    assert document is existing_document

    repository.create_document.assert_not_called()
    repository.commit.assert_not_called()


def test_register_document_rejects_missing_file():
    repository = MagicMock()

    service = DocumentRegistrationService(
        repository=repository
    )

    missing_path = (
        "data/documents/does-not-exist.pdf"
    )

    try:
        service.register_document(
            missing_path
        )
    except FileNotFoundError as error:
        assert (
            str(error)
            == "Document not found: "
            + missing_path
        )
    else:
        raise AssertionError(
            "Expected FileNotFoundError"
        )


def test_register_document_rejects_directory(
    tmp_path: Path,
):
    repository = MagicMock()

    service = DocumentRegistrationService(
        repository=repository
    )

    directory = tmp_path / "documents"
    directory.mkdir()

    try:
        service.register_document(
            str(directory)
        )
    except ValueError as error:
        assert (
            str(error)
            == f"Path is not a file: {directory}"
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )
