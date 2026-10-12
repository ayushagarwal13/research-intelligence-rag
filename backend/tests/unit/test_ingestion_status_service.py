from unittest.mock import Mock

import pytest

from app.services.ingestion_status_service import (
    IngestionStatusService,
)


def test_get_job_status_returns_existing_job():
    repository = Mock()

    repository.get_job_model.return_value = Mock(
        job_id="job-001",
        status="processing",
        total_documents=10,
        completed_documents=6,
        failed_documents=1,
    )

    service = IngestionStatusService(
        repository=repository
    )

    result = service.get_job_status(
        "job-001"
    )

    assert result.job_id == "job-001"
    assert result.status == "processing"
    assert result.total_documents == 10
    assert result.completed_documents == 6
    assert result.failed_documents == 1

    repository.get_job_model.assert_called_once_with(
        "job-001"
    )


def test_get_job_status_raises_for_missing_job():
    repository = Mock()

    repository.get_job_model.return_value = None

    service = IngestionStatusService(
        repository=repository
    )

    with pytest.raises(
        ValueError,
        match="Ingestion job not found",
    ):
        service.get_job_status(
            "does-not-exist"
        )
