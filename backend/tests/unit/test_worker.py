from unittest.mock import MagicMock, patch

from app.worker import (
    POLL_INTERVAL_SECONDS,
    create_worker_runner,
)


def test_create_worker_runner():
    fake_worker = MagicMock()

    with patch(
        "app.worker.create_ingestion_worker",
        return_value=fake_worker,
    ):
        runner = create_worker_runner()

    assert runner.worker is fake_worker
    assert (
        runner.poll_interval_seconds
        == POLL_INTERVAL_SECONDS
    )


def test_main_starts_worker_runner():
    fake_runner = MagicMock()

    with patch(
        "app.worker.create_worker_runner",
        return_value=fake_runner,
    ):
        from app.worker import main

        main()

    fake_runner.run.assert_called_once_with()
