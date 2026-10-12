from app.database.document_repository import (
    DocumentRepository,
)
from app.database.ingestion_repository import (
    IngestionRepository,
)
from app.services.document_registration_service import (
    DocumentRegistrationService,
)
from app.services.ingestion_job_service import (
    IngestionJobService,
)


class IngestionSubmissionService:
    """
    Coordinate document registration and ingestion-job creation.
    """

    def __init__(
        self,
        document_repository: DocumentRepository,
        ingestion_repository: IngestionRepository,
    ):
        self.document_repository = document_repository

        self.document_registration_service = (
            DocumentRegistrationService(
                repository=document_repository
            )
        )

        self.ingestion_job_service = (
            IngestionJobService(
                repository=ingestion_repository
            )
        )

    def submit_documents(
        self,
        file_paths: list[str],
    ):
        """
        Register documents and create an ingestion job
        for the registered documents.
        """
        if not file_paths:
            raise ValueError(
                "At least one document is required."
            )

        registered_documents = []

        for file_path in file_paths:
            document = (
                self.document_registration_service
                .register_document(file_path)
            )

            registered_documents.append(
                document
            )

        document_names = [
            document.document_name
            for document in registered_documents
        ]

        document_ids = [
            document.document_id
            for document in registered_documents
        ]

        return self.ingestion_job_service.create_job(
            document_names=document_names,
            document_ids=document_ids,
        )

    def submit_registered_documents(
        self,
        document_ids: list[str],
    ):
        """
        Create an ingestion job for documents that
        have already been registered.
        """
        if not document_ids:
            raise ValueError(
                "At least one document is required."
            )

        if len(set(document_ids)) != len(document_ids):
            raise ValueError(
                "Duplicate document IDs are not allowed "
                "within the same job."
            )

        registered_documents = []

        for document_id in document_ids:
            document = (
                self.document_repository.get_document(
                    document_id
                )
            )

            if document is None:
                raise ValueError(
                    "Document not found: "
                    f"{document_id}"
                )

            registered_documents.append(
                document
            )

        document_names = [
            document.document_name
            for document in registered_documents
        ]

        return self.ingestion_job_service.create_job(
            document_names=document_names,
            document_ids=document_ids,
        )
