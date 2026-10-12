import time

from app.database.ingestion_repository import (
    IngestionRepository,
)
from app.database.session import SessionLocal
from app.services.ingestion_worker import (
    IngestionWorker,
)


class WorkerRunner:
    """
    Continuously processes queued ingestion documents.

    The runner creates a fresh database session for each
    processing cycle so that long-running workers do not
    keep one database session open indefinitely.
    """

    def __init__(
        self,
        worker: IngestionWorker,
        poll_interval_seconds: int = 5,
    ):
        if poll_interval_seconds <= 0:
            raise ValueError(
                "poll_interval_seconds must be greater than zero."
            )

        self.worker = worker
        self.poll_interval_seconds = (
            poll_interval_seconds
        )

    def process_once(self) -> int:
        """
        Process available queued documents once.

        Returns:
            Number of documents processed.
        """
        processed_count = 0

        with SessionLocal() as session:
            repository = IngestionRepository(
                session=session
            )

            job_ids = self._get_queued_job_ids(
                repository
            )

            for job_id in job_ids:
                processed_count += (
                    self.worker.process_queued_documents(
                        repository=repository,
                        job_id=job_id,
                    )
                )

        return processed_count

    def run(self) -> None:
        """
        Continuously poll for queued work.
        """
        while True:
            processed_count = self.process_once()

            if processed_count == 0:
                time.sleep(
                    self.poll_interval_seconds
                )

    @staticmethod
    def _get_queued_job_ids(
        repository: IngestionRepository,
    ) -> list[str]:
        jobs = repository.get_queued_jobs()

        return [
            job.job_id
            for job in jobs
        ]
