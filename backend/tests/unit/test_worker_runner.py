from app.services.worker_runner import WorkerRunner
import pytest
from unittest.mock import MagicMock, patch


def test_process_once_processes_queued_jobs():
    worker = MagicMock()
    repository = MagicMock()

    queued_job_1 = MagicMock()
    queued_job_1.job_id = "job-001"

    queued_job_2 = MagicMock()
    queued_job_2.job_id = "job-002"

    repository.get_queued_jobs.return_value = [
        queued_job_1,
        queued_job_2,
    ]

    worker.process_queued_documents.side_effect = [
        2,
        3,
    ]

    with patch(
        "app.services.worker_runner.SessionLocal"
    ) as session_local:
        with patch(
            "app.services.worker_runner.IngestionRepository",
            return_value=repository,
        ):
            runner = WorkerRunner(
                worker=worker,
                poll_interval_seconds=1,
            )

            processed_count = runner.process_once()

    assert processed_count == 5

    repository.get_queued_jobs.assert_called_once_with()

    assert (
        worker.process_queued_documents.call_count
        == 2
    )

    worker.process_queued_documents.assert_any_call(
        repository=repository,
        job_id="job-001",
    )

    worker.process_queued_documents.assert_any_call(
        repository=repository,
        job_id="job-002",
    )

    session_local.assert_called_once_with()


def test_process_once_returns_zero_when_no_jobs_are_queued():
    worker = MagicMock()
    repository = MagicMock()
    repository.get_queued_jobs.return_value = []

    with patch(
        "app.services.worker_runner.SessionLocal"
    ):
        with patch(
            "app.services.worker_runner.IngestionRepository",
            return_value=repository,
        ):
            runner = WorkerRunner(
                worker=worker,
                poll_interval_seconds=1,
            )

            processed_count = runner.process_once()

    assert processed_count == 0

    repository.get_queued_jobs.assert_called_once_with()

    worker.process_queued_documents.assert_not_called()


@pytest.mark.parametrize(
    "interval",
    [0, -1, -5],
)
def test_worker_runner_rejects_invalid_poll_interval(interval):
    with pytest.raises(
        ValueError,
        match="poll_interval_seconds must be greater than zero.",
    ):
        WorkerRunner(
            worker=MagicMock(),
            poll_interval_seconds=interval,
        )


def test_run_sleeps_when_no_documents_are_processed():
    runner = WorkerRunner(
        worker=MagicMock(),
        poll_interval_seconds=3,
    )

    with patch.object(
        runner,
        "process_once",
        side_effect=[0, KeyboardInterrupt],
    ) as process_once:
        with patch(
            "app.services.worker_runner.time.sleep"
        ) as sleep:
            with pytest.raises(KeyboardInterrupt):
                runner.run()

    assert process_once.call_count == 2
    sleep.assert_called_once_with(3)


def test_run_does_not_sleep_when_documents_are_processed():
    runner = WorkerRunner(
        worker=MagicMock(),
        poll_interval_seconds=3,
    )

    with patch.object(
        runner,
        "process_once",
        side_effect=[2, KeyboardInterrupt],
    ) as process_once:
        with patch(
            "app.services.worker_runner.time.sleep"
        ) as sleep:
            with pytest.raises(KeyboardInterrupt):
                runner.run()

    assert process_once.call_count == 2
    sleep.assert_not_called()


