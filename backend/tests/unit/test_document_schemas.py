import pytest
from pydantic import ValidationError

from app.schemas.documents import (
    RegisterDocumentRequest,
    RegisterDocumentResponse,
)


def test_register_document_request_accepts_valid_path():
    request = RegisterDocumentRequest(
        file_path="data/documents/paper.pdf"
    )

    assert (
        request.file_path
        == "data/documents/paper.pdf"
    )


def test_register_document_request_rejects_empty_path():
    with pytest.raises(ValidationError):
        RegisterDocumentRequest(
            file_path=""
        )


def test_register_document_response_accepts_valid_data():
    response = RegisterDocumentResponse(
        document_id="document-001",
        document_name="paper.pdf",
        file_type=".pdf",
        page_count=12,
        status="registered",
    )

    assert (
        response.document_id
        == "document-001"
    )

    assert (
        response.document_name
        == "paper.pdf"
    )

    assert (
        response.page_count
        == 12
    )

    assert (
        response.status
        == "registered"
    )
