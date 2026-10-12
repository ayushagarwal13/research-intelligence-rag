from sqlalchemy.orm import Session

from app.database.models import DocumentModel


class DocumentRepository:
    """
    Repository responsible for persistence and retrieval
    of canonical document records.
    """

    def __init__(self, session: Session):
        self.session = session

    def create_document(
        self,
        document: DocumentModel,
    ) -> None:
        """
        Persist a new document record.
        """
        self.session.add(document)

    def get_document(
        self,
        document_id: str,
    ) -> DocumentModel | None:
        """
        Retrieve a document by its canonical document ID.
        """
        return self.session.get(
            DocumentModel,
            document_id,
        )

    def document_exists(
        self,
        document_id: str,
    ) -> bool:
        """
        Check whether a document already exists.
        """
        return (
            self.get_document(document_id)
            is not None
        )

    def update_document(
        self,
        document: DocumentModel,
    ) -> None:
        """
        Update an existing document record.
        """
        existing_document = self.get_document(
            document.document_id
        )

        if existing_document is None:
            raise ValueError(
                "Document not found: "
                f"{document.document_id}"
            )

        existing_document.document_name = (
            document.document_name
        )
        existing_document.file_type = (
            document.file_type
        )
        existing_document.document_path = (
            document.document_path
        )
        existing_document.document_content = (
            document.document_content
        )
        existing_document.page_count = (
            document.page_count
        )
        existing_document.status = (
            document.status
        )
        existing_document.updated_at = (
            document.updated_at
        )

        self.session.add(existing_document)

    def commit(self) -> None:
        """
        Commit the current transaction.
        """
        self.session.commit()
