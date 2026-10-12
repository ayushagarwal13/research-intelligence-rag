from app.database.qdrant_client import (
    create_collection,
    create_qdrant_client,
)
from app.embeddings.embedder import BGEEmbedder
from app.services.ingestion_worker import IngestionWorker


DEFAULT_COLLECTION_NAME = "research_documents"
DEFAULT_DOCUMENT_DIRECTORY = "data/documents"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50


def create_ingestion_worker() -> IngestionWorker:
    """
    Create a fully configured ingestion worker
    for the application.

    Database sessions are intentionally not created here.
    WorkerRunner owns the database session lifecycle and
    passes the active session to the repositories used
    during each processing cycle.
    """
    qdrant_client = create_qdrant_client()

    create_collection(
        client=qdrant_client,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    embedder = BGEEmbedder()

    return IngestionWorker(
        ingestion_job_service=None,
        qdrant_client=qdrant_client,
        embedder=embedder,
        collection_name=DEFAULT_COLLECTION_NAME,
        document_directory=DEFAULT_DOCUMENT_DIRECTORY,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_OVERLAP,
    )
