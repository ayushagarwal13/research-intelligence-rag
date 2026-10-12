
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"
ENV_FILE = PROJECT_ROOT / ".env"
VECTOR_SIZE = 768


def _load_qdrant_settings() -> tuple[str | None, str | None, str]:
    """Load environment settings before creating a Qdrant client."""
    load_dotenv(ENV_FILE, override=False)

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_path = os.getenv(
        "QDRANT_PATH",
        str(DEFAULT_QDRANT_PATH),
    )

    return qdrant_url, qdrant_api_key, qdrant_path


def create_qdrant_client() -> QdrantClient:
    """Connect to Qdrant Cloud or use local persistent storage."""
    qdrant_url, qdrant_api_key, qdrant_path = _load_qdrant_settings()

    if qdrant_url:
        return QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

    Path(qdrant_path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=qdrant_path)


def create_collection(
    client: QdrantClient,
    collection_name: str,
) -> None:
    """Create a collection if it does not already exist."""
    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )


def _create_point(
    chunk: dict,
    embedding: list[float],
) -> PointStruct:
    """Convert a chunk and embedding into a Qdrant point."""
    point_id = str(uuid5(NAMESPACE_URL, chunk["chunk_id"]))

    return PointStruct(
        id=point_id,
        vector=embedding,
        payload={
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
        },
    )


def insert_chunk(
    client: QdrantClient,
    collection_name: str,
    chunk: dict,
    embedding: list[float],
) -> None:
    """Insert one document chunk into Qdrant."""
    client.upsert(
        collection_name=collection_name,
        points=[_create_point(chunk, embedding)],
    )


def insert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """Insert multiple document chunks into Qdrant."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks must match number of embeddings."
        )

    if not chunks:
        return

    points = [
        _create_point(chunk, embedding)
        for chunk, embedding in zip(chunks, embeddings)
    ]

    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )
