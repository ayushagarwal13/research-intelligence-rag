
from dotenv import load_dotenv
from pathlib import Path
import logging

from app.services.worker_dependencies import create_ingestion_worker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 5


def create_worker_runner():
    logger.info("Initializing ingestion worker dependencies...")

    from app.services.worker_runner import WorkerRunner

    worker = create_ingestion_worker()
    logger.info("Ingestion worker dependencies initialized.")

    return WorkerRunner(
        worker=worker,
        poll_interval_seconds=POLL_INTERVAL_SECONDS,
    )


def main() -> None:
    logger.info("Starting Research Intelligence ingestion worker.")

    try:
        runner = create_worker_runner()
        logger.info("Worker initialized. Beginning job polling.")
        runner.run()
    except Exception:
        logger.exception("Ingestion worker stopped because of an error.")
        raise


if __name__ == "__main__":
    main()
