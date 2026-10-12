from app.database.ingestion_repository import (
    IngestionRepository,
)


class IngestionStatusService:
    """
    Service responsible for retrieving the current
    status of an ingestion job.
    """

    def __init__(
        self,
        repository: IngestionRepository,
    ):
        self.repository = repository

    def get_job_status(
        self,
        job_id: str,
    ):
        """
        Retrieve the current persisted state of an
        ingestion job.

        Raises:
            ValueError: If the job does not exist.
        """
        job = self.repository.get_job_model(
            job_id
        )

        if job is None:
            raise ValueError(
                f"Ingestion job not found: {job_id}"
            )

        return job
