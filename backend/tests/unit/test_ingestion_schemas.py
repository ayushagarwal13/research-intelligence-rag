import pytest
from pydantic import ValidationError

from app.schemas.ingestion import (
    CreateIngestionJobRequest,
    RegisteredDocumentsIngestionRequest,
)


def test_registered_documents_request_accepts_valid_ids():
    request = RegisteredDocumentsIngestionRequest(
        document_ids=[
            "document-001",
            "document-002",
        ]
    )

    assert request.document_ids == [
        "document-001",
        "document-002",
    ]


def test_registered_documents_request_rejects_empty_list():
    with pytest.raises(ValidationError):
        RegisteredDocumentsIngestionRequest(
            document_ids=[]
        )


def test_registered_documents_request_rejects_empty_document_id():
    with pytest.raises(ValidationError):
        RegisteredDocumentsIngestionRequest(
            document_ids=[
                "document-001",
                "",
            ]
        )


def test_existing_ingestion_request_still_works():
    request = CreateIngestionJobRequest(
        documents=[
            {
                "document_id": "document-001",
                "document_name": "paper.pdf",
            }
        ]
    )

    assert len(request.documents) == 1
    assert (
        request.documents[0].document_id
        == "document-001"
    )
