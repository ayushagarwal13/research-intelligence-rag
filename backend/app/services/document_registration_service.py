from pathlib import Path

import pymupdf

from app.database.document_repository import (
    DocumentRepository,
)
from app.database.models import DocumentModel
from app.ingestion.document_identity import (
    generate_document_id,
)
from app.ingestion.pdf_loader import (
    extract_text_from_pdf,
)


MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024


class DocumentRegistrationService:
    """
    Register physical documents in the application's
    canonical document registry.

    Documents may be registered from:
    - an existing filesystem path, or
    - uploaded PDF bytes.
    """

    def __init__(
        self,
        repository: DocumentRepository,
    ):
        self.repository = repository

    def register_document(
        self,
        file_path: str,
    ) -> DocumentModel:
        """
        Register a physical PDF document.

        The document ID is derived from the file contents,
        making the identity deterministic.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {file_path}"
            )

        extracted_document = (
            extract_text_from_pdf(
                str(path)
            )
        )

        document_id = (
            extracted_document["document_id"]
        )

        existing_document = (
            self.repository.get_document(
                document_id
            )
        )

        if existing_document is not None:
            return existing_document

        document = DocumentModel(
            document_id=document_id,
            document_name=(
                extracted_document[
                    "document_name"
                ]
            ),
            file_type=(
                extracted_document[
                    "file_type"
                ]
            ),
            document_path=(
                extracted_document[
                    "document_path"
                ]
            ),
            document_content=None,
            page_count=(
                extracted_document[
                    "page_count"
                ]
            ),
            status="registered",
        )

        self.repository.create_document(
            document
        )
        self.repository.commit()

        return document

    def register_uploaded_document(
        self,
        file_name: str,
        file_content: bytes,
    ) -> DocumentModel:
        """
        Register a PDF received through an HTTP upload.

        The uploaded bytes are stored directly in the
        database so the API and worker do not need to
        share a filesystem.
        """
        normalized_file_name = (
            Path(file_name).name
        )

        if not normalized_file_name:
            raise ValueError(
                "Uploaded file must have a valid name."
            )

        if not normalized_file_name.lower().endswith(
            ".pdf"
        ):
            raise ValueError(
                "Only PDF files are supported."
            )

        if not file_content:
            raise ValueError(
                "Uploaded file is empty."
            )

        if len(file_content) > MAX_UPLOAD_SIZE_BYTES:
            raise ValueError(
                "Uploaded PDF exceeds the 20 MB size limit."
            )

        if not file_content.startswith(b"%PDF"):
            raise ValueError(
                "Uploaded file is not a valid PDF."
            )

        document_id = generate_document_id_from_bytes(
            file_content
        )

        existing_document = (
            self.repository.get_document(
                document_id
            )
        )

        if existing_document is not None:
            return existing_document

        page_count = self._get_pdf_page_count(
            file_content
        )

        document = DocumentModel(
            document_id=document_id,
            document_name=normalized_file_name,
            file_type=".pdf",
            document_path=None,
            document_content=file_content,
            page_count=page_count,
            status="registered",
        )

        self.repository.create_document(
            document
        )
        self.repository.commit()

        return document

    @staticmethod
    def _get_pdf_page_count(
        file_content: bytes,
    ) -> int:
        """
        Validate that the uploaded bytes can be opened
        as a PDF and return the page count.
        """
        try:
            pdf_document = pymupdf.open(
                stream=file_content,
                filetype="pdf",
            )
        except Exception as error:
            raise ValueError(
                "Uploaded file could not be opened as a PDF."
            ) from error

        try:
            page_count = pdf_document.page_count
        finally:
            pdf_document.close()

        if page_count <= 0:
            raise ValueError(
                "Uploaded PDF contains no pages."
            )

        return page_count


def generate_document_id_from_bytes(
    file_content: bytes,
) -> str:
    """
    Generate the canonical document ID from PDF bytes.

    The ID is the SHA-256 hash of the exact uploaded
    document contents.
    """
    import hashlib

    return hashlib.sha256(
        file_content
    ).hexdigest()
