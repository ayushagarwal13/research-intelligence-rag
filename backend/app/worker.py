from app.services.worker_runner import (
    WorkerRunner,
)
from app.services.worker_dependencies import (
    create_ingestion_worker,
)
from dotenv import load_dotenv
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


POLL_INTERVAL_SECONDS = 5


def create_worker_runner() -> WorkerRunner:
    """Create a fully configured worker runner."""
    worker = create_ingestion_worker()

    return WorkerRunner(
        worker=worker,
        poll_interval_seconds=POLL_INTERVAL_SECONDS,
    )


def main() -> None:
    """Start the ingestion worker process."""
    runner = create_worker_runner()
    runner.run()


if __name__ == "__main__":
    main()
