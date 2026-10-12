from pathlib import Path

import pymupdf
from sqlalchemy.orm import Session

from app.database.models import DocumentModel


def create_test_pdf(file_path: Path) -> None:
    document = pymupdf.open()

    page = document.new_page()

    page.insert_text(
        (72, 72),
        "This is a research document "
        "used for API testing.",
    )

    document.save(file_path)
    document.close()


def test_register_document_endpoint(
    tmp_path: Path,
    api_client,
    test_database,
):
    document_path = tmp_path / "paper.pdf"
    create_test_pdf(document_path)

    response = api_client.post(
        "/documents/register",
        json={
            "file_path": str(document_path),
        },
    )

    assert response.status_code == 201

    response_data = response.json()

    assert response_data["document_name"] == "paper.pdf"
    assert response_data["file_type"] == ".pdf"
    assert response_data["page_count"] == 1
    assert response_data["status"] == "registered"

    with test_database() as db:
        document = db.get(
            DocumentModel,
            response_data["document_id"],
        )

        assert document is not None
        assert document.document_name == "paper.pdf"
        assert document.status == "registered"


def test_register_document_endpoint_returns_404_for_missing_file(
    api_client,
):
    missing_file_path = (
        "data/documents/does-not-exist.pdf"
    )

    response = api_client.post(
        "/documents/register",
        json={
            "file_path": missing_file_path,
        },
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": (
            "Document not found: "
            "data/documents/"
            "does-not-exist.pdf"
        )
    }
