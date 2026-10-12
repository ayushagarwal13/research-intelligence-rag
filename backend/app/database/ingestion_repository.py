from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import (
    DocumentModel,
    IngestionDocumentModel,
    IngestionJobModel,
    utc_now,
)
from app.models.ingestion import (
    MAX_INGESTION_ATTEMPTS,
    IngestionDocument,
    IngestionJob,
)


class IngestionRepository:
    """
    Repository responsible for persistence and retrieval of
    ingestion jobs and ingestion documents.

    Application-level CRUD methods work with dataclasses.

    Database-worker methods work directly with SQLAlchemy
    ORM models.
    """

    def __init__(self, session: Session):
        self.session = session

    # ------------------------------------------------------------------
    # Job CRUD
    # ------------------------------------------------------------------

    def create_job(self, job: IngestionJob) -> None:
        job_model = IngestionJobModel(
            job_id=job.job_id,
            total_documents=job.total_documents,
            completed_documents=job.completed_documents,
            failed_documents=job.failed_documents,
            status=job.status,
            created_at=job.created_at or utc_now(),
            updated_at=job.updated_at or utc_now(),
        )

        self.session.add(job_model)

    def get_job(self, job_id: str) -> IngestionJob | None:
        job_model = self.session.get(
            IngestionJobModel,
            job_id,
        )

        if job_model is None:
            return None

        return IngestionJob(
            job_id=job_model.job_id,
            total_documents=job_model.total_documents,
            completed_documents=job_model.completed_documents,
            failed_documents=job_model.failed_documents,
            status=job_model.status,
            created_at=job_model.created_at,
            updated_at=job_model.updated_at,
        )

    def get_job_model(
        self,
        job_id: str,
    ) -> IngestionJobModel | None:
        """
        Retrieve the SQLAlchemy job model.

        This method is used by application layers that
        need the current persisted job state.
        """
        return self.session.get(
            IngestionJobModel,
            job_id,
        )

    def update_job(self, job: IngestionJob) -> None:
        job_model = self.session.get(
            IngestionJobModel,
            job.job_id,
        )

        if job_model is None:
            raise ValueError(
                f"Ingestion job not found: {job.job_id}"
            )

        job_model.total_documents = job.total_documents
        job_model.completed_documents = job.completed_documents
        job_model.failed_documents = job.failed_documents
        job_model.status = job.status
        job_model.updated_at = utc_now()

        self.session.add(job_model)

    # ------------------------------------------------------------------
    # Ingestion document CRUD
    # ------------------------------------------------------------------

    def create_ingestion_document(
        self,
        document: IngestionDocument,
    ) -> None:
        document_model = IngestionDocumentModel(
            job_id=document.job_id,
            document_id=document.document_id,
            document_name=document.document_name,
            status=document.status,
            error_message=document.error_message,
            attempt_count=document.attempt_count,
            created_at=utc_now(),
            updated_at=utc_now(),
        )

        self.session.add(document_model)

    def create_ingestion_documents(
        self,
        documents: list[IngestionDocument],
    ) -> None:
        """
        Create multiple ingestion-document records.
        """
        for document in documents:
            self.create_ingestion_document(document)

    def get_ingestion_document(
        self,
        job_id: str,
        document_id: str,
    ) -> IngestionDocument | None:
        document_model = self.session.get(
            IngestionDocumentModel,
            {
                "job_id": job_id,
                "document_id": document_id,
            },
        )

        if document_model is None:
            return None

        return IngestionDocument(
            document_id=document_model.document_id,
            job_id=document_model.job_id,
            document_name=document_model.document_name,
            status=document_model.status,
            error_message=document_model.error_message,
            attempt_count=document_model.attempt_count,
        )

    def update_ingestion_document(
        self,
        document: IngestionDocument,
    ) -> None:
        document_model = self.session.get(
            IngestionDocumentModel,
            {
                "job_id": document.job_id,
                "document_id": document.document_id,
            },
        )

        if document_model is None:
            raise ValueError(
                "Ingestion document not found: "
                f"{document.job_id}/{document.document_id}"
            )

        document_model.document_name = document.document_name
        document_model.status = document.status
        document_model.error_message = document.error_message
        document_model.attempt_count = document.attempt_count
        document_model.updated_at = utc_now()

        self.session.add(document_model)

    def update_document(
        self,
        document: IngestionDocument,
    ) -> None:
        self.update_ingestion_document(document)

    # ------------------------------------------------------------------
    # Physical document persistence
    # ------------------------------------------------------------------

    def create_document(
        self,
        document: DocumentModel,
    ) -> None:
        """
        Create a canonical document record.

        This method is used when registering a document
        before creating an ingestion job.
        """
        self.create_document_record(document)

    def create_document_record(
        self,
        document: DocumentModel,
    ) -> None:
        self.session.add(document)

    def get_document_record(
        self,
        document_id: str,
    ) -> DocumentModel | None:
        return self.session.get(
            DocumentModel,
            document_id,
        )

    # ------------------------------------------------------------------
    # Worker job discovery
    # ------------------------------------------------------------------

    def get_queued_jobs(
        self,
    ) -> list[IngestionJobModel]:
        """
        Return ingestion jobs that currently have
        at least one queued document.

        Jobs are ordered by creation time so older
        jobs are discovered before newer jobs.
        """
        statement = (
            select(IngestionJobModel)
            .join(
                IngestionDocumentModel,
                IngestionDocumentModel.job_id
                == IngestionJobModel.job_id,
            )
            .where(
                IngestionDocumentModel.status == "queued",
                IngestionJobModel.status.in_(
                    ["queued", "processing"]
                ),
            )
            .distinct()
            .order_by(
                IngestionJobModel.created_at
            )
        )

        return list(
            self.session.scalars(
                statement
            ).all()
        )

    # ------------------------------------------------------------------
    # Database-backed worker operations
    # ------------------------------------------------------------------

    def get_documents_for_job(
        self,
        job_id: str,
    ) -> list[IngestionDocumentModel]:
        statement = (
            select(IngestionDocumentModel)
            .where(
                IngestionDocumentModel.job_id
                == job_id
            )
            .order_by(
                IngestionDocumentModel.created_at
            )
        )

        return list(
            self.session.scalars(
                statement
            ).all()
        )

    def update_document_model(
        self,
        document: IngestionDocumentModel,
    ) -> None:
        document.updated_at = utc_now()
        self.session.add(document)

    def update_job_model(
        self,
        job: IngestionJobModel,
    ) -> None:
        job.updated_at = utc_now()
        self.session.add(job)

    def claim_next_queued_document(
        self,
        job_id: str,
    ) -> IngestionDocumentModel | None:
        statement = (
            select(IngestionDocumentModel)
            .where(
                IngestionDocumentModel.job_id
                == job_id,
                IngestionDocumentModel.status
                == "queued",
                IngestionDocumentModel.attempt_count
                < MAX_INGESTION_ATTEMPTS,
            )
            .order_by(
                IngestionDocumentModel.created_at
            )
            .limit(1)
            .with_for_update(
                skip_locked=True
            )
        )

        document = self.session.scalars(
            statement
        ).first()

        if document is None:
            return None

        document.status = "processing"
        document.attempt_count += 1
        document.error_message = None
        document.updated_at = utc_now()

        self.session.add(document)

        return document

    def update_job_progress(
        self,
        job_id: str,
    ) -> IngestionJobModel:
        job = self.session.get(
            IngestionJobModel,
            job_id,
        )

        if job is None:
            raise ValueError(
                f"Ingestion job not found: {job_id}"
            )

        documents = self.get_documents_for_job(
            job_id
        )

        completed_count = sum(
            document.status == "completed"
            for document in documents
        )

        failed_count = sum(
            document.status == "failed"
            for document in documents
        )

        processing_count = sum(
            document.status == "processing"
            for document in documents
        )

        queued_count = sum(
            document.status == "queued"
            for document in documents
        )

        job.completed_documents = completed_count
        job.failed_documents = failed_count

        if (
            completed_count + failed_count
            == job.total_documents
        ):
            if failed_count > 0:
                job.status = "completed_with_errors"
            else:
                job.status = "completed"

        elif processing_count > 0:
            job.status = "processing"

        elif queued_count > 0:
            if (
                completed_count > 0
                or failed_count > 0
            ):
                job.status = "processing"
            else:
                job.status = "queued"

        else:
            job.status = "processing"

        job.updated_at = utc_now()

        self.session.add(job)

        return job

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def recover_stale_processing_documents(
        self,
        timeout_minutes: int = 30,
    ) -> int:
        if timeout_minutes <= 0:
            raise ValueError(
                "timeout_minutes must be greater than zero."
            )

        cutoff_time = (
            datetime.now(timezone.utc)
            - timedelta(
                minutes=timeout_minutes
            )
        )

        statement = select(
            IngestionDocumentModel
        ).where(
            IngestionDocumentModel.status
            == "processing",
            IngestionDocumentModel.updated_at
            < cutoff_time,
        )

        stale_documents = list(
            self.session.scalars(
                statement
            ).all()
        )

        for document in stale_documents:
            document.status = "queued"
            document.error_message = (
                "Recovered after worker timeout."
            )
            document.updated_at = utc_now()

        return len(stale_documents)

    # ------------------------------------------------------------------
    # Transaction
    # ------------------------------------------------------------------

    def commit(self) -> None:
        self.session.commit()
