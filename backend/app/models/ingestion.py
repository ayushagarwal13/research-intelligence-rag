from dataclasses import dataclass
from datetime import datetime


JOB_STATUSES = {
    "queued",
    "processing",
    "completed",
    "completed_with_errors",
}


DOCUMENT_STATUSES = {
    "queued",
    "processing",
    "completed",
    "failed",
}


MAX_INGESTION_ATTEMPTS = 3


@dataclass
class IngestionJob:
    """
    Represents one document ingestion request.
    """

    job_id: str
    total_documents: int
    completed_documents: int = 0
    failed_documents: int = 0
    status: str = "queued"
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class IngestionDocument:
    """
    Represents one document inside an ingestion job.
    """

    document_id: str
    job_id: str
    document_name: str
    status: str = "queued"
    error_message: str | None = None
    attempt_count: int = 0
