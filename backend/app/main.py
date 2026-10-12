from app.database.session import engine
from app.database.models import (
    DocumentModel,
    IngestionJobModel,
    IngestionDocumentModel,
    DocumentChunkModel,
)
from app.database.base import Base
from app.api.query import router as query_router
from app.api.ingestion import router as ingestion_router
from app.api.documents import router as documents_router
from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables before importing application modules.
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the database and shared runtime when the application starts.
    """
    _ = (
        DocumentModel,
        IngestionJobModel,
        IngestionDocumentModel,
        DocumentChunkModel,
    )

    Base.metadata.create_all(bind=engine)
    app.state.container = None

    try:
        yield
    finally:
        container = getattr(app.state, "container", None)

        if container is not None:
            container.qdrant_client.close()


app = FastAPI(
    title="Research Intelligence RAG",
    description=(
        "Research paper ingestion and "
        "retrieval-augmented generation backend."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(documents_router)
app.include_router(ingestion_router)
app.include_router(query_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

