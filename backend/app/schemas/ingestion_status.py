from pydantic import BaseModel


class IngestionJobStatusResponse(BaseModel):
    """
    Response containing the current state of an ingestion job.
    """

    job_id: str
    status: str
    total_documents: int
    completed_documents: int
    failed_documents: int
