from unittest.mock import Mock
import uuid

from app.database.ingestion_repository import (
    IngestionRepository,
)
from app.models.ingestion import (
    MAX_INGESTION_ATTEMPTS,
    IngestionDocument,
    IngestionJob,
)


class IngestionJobService:
    """
    Handles ingestion business logic and persistence.

    The production ingestion flow should provide canonical
    document IDs. The optional document_ids argument is kept
    for backward compatibility with the original service API
    and existing unit tests.
    """

    def __init__(
        self,
        repository: IngestionRepository,
    ):
        self.repository = repository

    def create_job(
        self,
        document_names: list[str],
        document_ids: list[str] | None = None,
    ) -> tuple[
        IngestionJob,
        list[IngestionDocument],
    ]:
        """
        Create an ingestion job.

        Args:
            document_names:
                Names of the documents in the job.

            document_ids:
                Canonical document IDs corresponding one-to-one
                with document_names.

                When omitted, deterministic compatibility IDs
                are generated from the document names. The real
                production ingestion flow should always provide
                canonical document IDs generated from document
                contents.
        """

        if not document_names:
            raise ValueError(
                "At least one document is required."
            )

        if document_ids is None:
            document_ids = [
                str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"legacy-document:{document_name}",
                    )
                )
                for document_name in document_names
            ]

        if not document_ids:
            raise ValueError(
                "At least one document ID is required."
            )

        if len(document_names) != len(document_ids):
            raise ValueError(
                "Number of document names must match "
                "number of document IDs."
            )

        if len(set(document_ids)) != len(document_ids):
            raise ValueError(
                "Duplicate document IDs are not "
                "allowed within the same job."
            )

        for document_id in document_ids:
            if not document_id.strip():
                raise ValueError(
                    "Document IDs cannot be empty."
                )

        job = IngestionJob(
            job_id=str(uuid.uuid4()),
            total_documents=len(document_names),
        )

        documents = [
            IngestionDocument(
                document_id=document_id,
                job_id=job.job_id,
                document_name=document_name,
            )
            for document_name, document_id in zip(
                document_names,
                document_ids,
            )
        ]

        self.repository.create_job(job)

        self._create_ingestion_documents(documents)

        self.repository.commit()

        return job, documents

    def _create_ingestion_documents(
        self,
        documents: list[IngestionDocument],
    ) -> None:
        """
        Persist ingestion documents.

        The compatibility fallback supports the older
        in-memory test repository while the production
        repository uses create_ingestion_documents().
        """

        create_method = getattr(
            self.repository,
            "create_ingestion_documents",
            None,
        )

        if create_method is not None:
            create_method(documents)
            return

        legacy_create_method = getattr(
            self.repository,
            "create_documents",
            None,
        )

        if legacy_create_method is not None:
            legacy_create_method(documents)
            return

        raise AttributeError(
            "Repository must provide either "
            "create_ingestion_documents() or "
            "create_documents()."
        )

    def mark_document_processing(
        self,
        document: IngestionDocument,
    ) -> None:
        """
        Move a queued document into processing.
        """

        if document.status != "queued":
            raise ValueError(
                "Only queued documents can start processing."
            )

        if (
            document.attempt_count
            >= MAX_INGESTION_ATTEMPTS
        ):
            raise ValueError(
                "Maximum ingestion attempts exceeded."
            )

        document.status = "processing"
        document.attempt_count += 1

        self.repository.update_document(
            document
        )

        self.repository.commit()

    def mark_document_completed(
        self,
        document: IngestionDocument,
        job: IngestionJob,
        documents: list[IngestionDocument],
    ) -> None:
        """
        Mark a document as completed and update
        job progress.
        """

        if document.status != "processing":
            raise ValueError(
                "Only processing documents can be completed."
            )

        document.status = "completed"
        document.error_message = None

        self._update_job_progress(
            job=job,
            documents=documents,
        )

        self.repository.update_document(
            document
        )

        self.repository.update_job(job)

        self.repository.commit()

    def mark_document_failed(
        self,
        document: IngestionDocument,
        job: IngestionJob,
        documents: list[IngestionDocument],
        error_message: str,
    ) -> None:
        """
        Mark a document as failed and update
        job progress.
        """

        if document.status != "processing":
            raise ValueError(
                "Only processing documents can fail."
            )

        if not error_message.strip():
            raise ValueError(
                "Error message cannot be empty."
            )

        document.status = "failed"
        document.error_message = error_message

        self._update_job_progress(
            job=job,
            documents=documents,
        )

        self.repository.update_document(
            document
        )

        self.repository.update_job(job)

        self.repository.commit()

    def retry_document(
        self,
        document: IngestionDocument,
        job: IngestionJob | None = None,
        documents: list[IngestionDocument] | None = None,
    ) -> None:
        """
        Move a failed document back to queued so that
        another processing attempt can occur.

        When job and documents are supplied, recalculate
        the job progress after moving the document to
        queued.
        """

        if document.status != "failed":
            raise ValueError(
                "Only failed documents can be retried."
            )

        if (
            document.attempt_count
            >= MAX_INGESTION_ATTEMPTS
        ):
            raise ValueError(
                "Maximum ingestion attempts exceeded."
            )

        document.status = "queued"
        document.error_message = None

        self.repository.update_document(
            document
        )

        if (
            job is not None
            and documents is not None
        ):
            self._update_job_progress(
                job=job,
                documents=documents,
            )

            self.repository.update_job(job)

        self.repository.commit()

    @staticmethod
    def _update_job_progress(
        job: IngestionJob,
        documents: list[IngestionDocument],
    ) -> None:
        """
        Calculate the current job progress.

        This in-memory calculation is suitable for
        the current application layer. The database-backed
        repository remains the authoritative source for
        persisted worker progress.
        """

        completed = sum(
            document.status == "completed"
            for document in documents
        )

        failed = sum(
            document.status == "failed"
            for document in documents
        )

        processing = sum(
            document.status == "processing"
            for document in documents
        )

        queued = sum(
            document.status == "queued"
            for document in documents
        )

        job.completed_documents = completed
        job.failed_documents = failed

        if completed == job.total_documents:
            job.status = "completed"

        elif (
            completed + failed
            == job.total_documents
            and failed > 0
        ):
            job.status = "completed_with_errors"

        elif processing > 0:
            job.status = "processing"

        elif (
            queued > 0
            and completed + failed > 0
        ):
            job.status = "processing"

        elif queued == job.total_documents:
            job.status = "queued"

        else:
            job.status = "processing"



