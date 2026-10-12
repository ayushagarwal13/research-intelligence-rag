
from pathlib import Path
from tempfile import NamedTemporaryFile

from app.services.ingestion_job_service import IngestionJobService
from app.services.document_indexing_service import index_document
from app.ingestion.pdf_loader import extract_text_from_pdf
from app.database.chunk_repository import ChunkRepository
from app.chunking.document_chunker import chunk_document


class IngestionWorker:
    """
    Processes ingestion documents through the complete
    ingestion pipeline.

    Supports:
    - Direct processing of known documents.
    - Database-backed claiming of queued documents.
    - Uploaded PDFs stored as database bytes.
    - Local filesystem PDFs for backward compatibility.
    - Database-authoritative job progress.
    - Stale processing recovery.
    - Persistent storage of document chunks.
    """

    def __init__(
        self,
        ingestion_job_service: IngestionJobService,
        qdrant_client,
        embedder,
        collection_name: str,
        document_directory: str = "data/documents",
        chunk_size: int = 500,
        overlap: int = 50,
        chunk_repository=None,
    ):
        self.ingestion_job_service = ingestion_job_service
        self.qdrant_client = qdrant_client
        self.embedder = embedder
        self.collection_name = collection_name
        self.document_directory = Path(document_directory)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunk_repository = chunk_repository

    def process_document(
        self,
        document,
        job,
        documents,
    ) -> None:
        """
        Process one document using the original
        in-memory job service flow.

        This path continues to use a physical file because
        it is retained for backward compatibility with the
        existing direct-processing workflow and tests.

        When a ChunkRepository is configured, the
        generated chunks are also persisted.
        """
        self.ingestion_job_service.mark_document_processing(
            document
        )

        try:
            document_path = self._resolve_document_path(
                document
            )

            extracted_document = extract_text_from_pdf(
                document_path
            )

            self._validate_document_identity(
                document=document,
                extracted_document=extracted_document,
            )

            index_document(
                client=self.qdrant_client,
                embedder=self.embedder,
                collection_name=self.collection_name,
                document=extracted_document,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )

            self._persist_chunks(
                extracted_document=extracted_document
            )

            self.ingestion_job_service.mark_document_completed(
                document=document,
                job=job,
                documents=documents,
            )

        except Exception as error:
            self.ingestion_job_service.mark_document_failed(
                document=document,
                job=job,
                documents=documents,
                error_message=str(error),
            )
            raise

    def process_job(
        self,
        job,
        documents,
    ) -> None:
        """
        Process all queued documents belonging to a job.
        """
        if not documents:
            raise ValueError(
                "At least one document is required."
            )

        for document in documents:
            if document.status != "queued":
                continue

            try:
                self.process_document(
                    document=document,
                    job=job,
                    documents=documents,
                )

            except Exception:
                continue

    def process_next_queued_document(
        self,
        repository,
        job_id: str,
    ) -> bool:
        """
        Claim and process one queued document from
        the database.

        Uploaded documents are read from the
        document_content column.

        Legacy documents that have no database content
        continue to use their local filesystem path.

        The database is authoritative for the document
        state and job progress.
        """
        document = repository.claim_next_queued_document(
            job_id=job_id
        )

        if document is None:
            return False

        try:
            extracted_document = (
                self._extract_database_document(
                    document
                )
            )

            self._validate_document_identity(
                document=document,
                extracted_document=extracted_document,
            )

            index_document(
                client=self.qdrant_client,
                embedder=self.embedder,
                collection_name=self.collection_name,
                document=extracted_document,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )

            if hasattr(repository, "session"):
                chunk_repository = ChunkRepository(
                    session=repository.session
                )

                self._persist_chunks(
                    extracted_document=extracted_document,
                    chunk_repository=chunk_repository,
                )

            document.status = "completed"
            document.error_message = None

        except Exception as error:
            document.status = "failed"
            document.error_message = str(error)

        repository.update_document_model(
            document
        )

        repository.update_job_progress(
            job_id=job_id
        )

        repository.commit()

        return True

    def process_queued_documents(
        self,
        repository,
        job_id: str,
    ) -> int:
        """
        Continuously claim queued documents until
        no eligible documents remain.

        Returns:
            Number of documents claimed.
        """
        processed_count = 0

        while True:
            claimed = self.process_next_queued_document(
                repository=repository,
                job_id=job_id,
            )

            if not claimed:
                break

            processed_count += 1

        return processed_count

    def retry_document(
        self,
        document,
        job,
        documents,
    ) -> None:
        """
        Retry a previously failed document.
        """
        if document.status != "failed":
            raise ValueError(
                "Only failed documents can be retried."
            )

        self.ingestion_job_service.retry_document(
            document
        )

        try:
            self.process_document(
                document=document,
                job=job,
                documents=documents,
            )

        except Exception:
            raise

    def retry_failed_documents(
        self,
        job,
        documents,
    ) -> None:
        """
        Retry every failed document in a job.
        """
        if not documents:
            raise ValueError(
                "At least one document is required."
            )

        for document in documents:
            if document.status != "failed":
                continue

            try:
                self.retry_document(
                    document=document,
                    job=job,
                    documents=documents,
                )

            except Exception:
                continue

    def recover_stale_documents(
        self,
        repository,
        timeout_minutes: int = 30,
    ) -> int:
        """
        Recover documents left in processing state
        after a worker crash or interruption.
        """
        recovered_count = (
            repository.recover_stale_processing_documents(
                timeout_minutes=timeout_minutes
            )
        )

        repository.commit()

        return recovered_count

    def _extract_database_document(
        self,
        document,
    ) -> dict:
        """
        Extract a registered document using database
        content when available.

        Uploaded documents are stored in
        document.document.document_content.

        Preserve the registered filename and document ID
        when extracting uploaded PDF bytes.

        Legacy documents, test doubles, and documents without
        stored bytes fall back to the configured local
        document directory.
        """
        document_model = getattr(
            document,
            "document",
            None,
        )

        if document_model is not None:
            document_content = getattr(
                document_model,
                "document_content",
                None,
            )

            if document_content:
                extracted_document = self._extract_from_bytes(
                    file_content=document_content
                )

                # Preserve the canonical metadata from the database.
                extracted_document["document_id"] = (
                    document.document_id
                )
                extracted_document["document_name"] = (
                    document_model.document_name
                )

                return extracted_document

        document_path = self._resolve_document_path(
            document
        )

        return extract_text_from_pdf(
            document_path
        )

    @staticmethod
    def _extract_from_bytes(
        file_content: bytes,
    ) -> dict:
        """
        Extract PDF text from bytes by temporarily
        materializing the PDF on disk.

        The temporary file is deleted immediately
        after extraction.
        """
        temporary_path = None

        try:
            with NamedTemporaryFile(
                suffix=".pdf",
                delete=False,
            ) as temporary_file:
                temporary_file.write(file_content)
                temporary_file.flush()

                temporary_path = Path(
                    temporary_file.name
                )

            return extract_text_from_pdf(
                str(temporary_path)
            )

        finally:
            if temporary_path is not None:
                temporary_path.unlink(
                    missing_ok=True
                )

    def _persist_chunks(
        self,
        extracted_document: dict,
        chunk_repository=None,
    ) -> None:
        """
        Persist the chunks produced from an extracted
        document.

        If chunk_repository is explicitly supplied, use it.

        Otherwise, fall back to the repository injected
        during worker construction. This preserves the
        existing direct-processing path and its tests.
        """
        repository = (
            chunk_repository
            or self.chunk_repository
        )

        if repository is None:
            return

        chunks = chunk_document(
            document=extracted_document,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

        repository.create_chunks(
            chunks
        )

    def _resolve_document_path(
        self,
        document,
    ) -> Path:
        """
        Resolve the physical document path from the
        configured document directory.
        """
        document_path = (
            self.document_directory
            / document.document_name
        )

        if not document_path.exists():
            raise FileNotFoundError(
                f"Document not found: {document_path}"
            )

        return document_path

    def _validate_document_identity(
        self,
        document,
        extracted_document: dict,
    ) -> None:
        """
        Ensure the physical PDF or uploaded PDF matches
        the document identity stored in the database.
        """
        extracted_document_id = (
            extracted_document["document_id"]
        )

        if (
            extracted_document_id
            != document.document_id
        ):
            raise ValueError(
                "Document identity mismatch. "
                "The physical file does not match "
                "the registered document ID."
            )
