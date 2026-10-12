from datetime import datetime, timedelta, timezone

from app.services.ingestion_worker import (
    IngestionWorker,
)


class FakeRepository:
    """
    Fake repository used to verify worker recovery
    orchestration.
    """

    def __init__(
        self,
        recovered_count=0,
    ):
        self.recovered_count = (
            recovered_count
        )
        self.timeout_minutes = None
        self.commit_called = False

    def recover_stale_processing_documents(
        self,
        timeout_minutes,
    ):
        self.timeout_minutes = (
            timeout_minutes
        )

        return self.recovered_count

    def commit(self):
        self.commit_called = True


def create_worker():
    """
    Create a worker with fake dependencies because
    recovery does not require Qdrant, embeddings, or PDFs.
    """

    return IngestionWorker(
        ingestion_job_service=None,
        qdrant_client=None,
        embedder=None,
        collection_name="test",
    )


def test_recover_stale_documents_returns_recovered_count():
    repository = FakeRepository(
        recovered_count=3
    )

    worker = create_worker()

    recovered_count = (
        worker.recover_stale_documents(
            repository=repository,
            timeout_minutes=30,
        )
    )

    assert recovered_count == 3

    assert (
        repository.timeout_minutes
        == 30
    )

    assert repository.commit_called is True


def test_recover_stale_documents_works_when_nothing_is_stale():
    repository = FakeRepository(
        recovered_count=0
    )

    worker = create_worker()

    recovered_count = (
        worker.recover_stale_documents(
            repository=repository,
            timeout_minutes=30,
        )
    )

    assert recovered_count == 0

    assert repository.commit_called is True


def test_recover_stale_documents_passes_custom_timeout():
    repository = FakeRepository(
        recovered_count=2
    )

    worker = create_worker()

    worker.recover_stale_documents(
        repository=repository,
        timeout_minutes=60,
    )

    assert (
        repository.timeout_minutes
        == 60
    )


def test_recover_stale_documents_commits_after_recovery():
    repository = FakeRepository(
        recovered_count=1
    )

    worker = create_worker()

    worker.recover_stale_documents(
        repository=repository,
        timeout_minutes=15,
    )

    assert repository.commit_called is True
