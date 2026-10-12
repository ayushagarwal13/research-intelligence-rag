
from qdrant_client.models import PointStruct
from app.embeddings.embedder import BGEEmbedder
from app.database.session import SessionLocal
from app.database.chunk_repository import ChunkRepository
import app.database.qdrant_client as qdrant_module
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = PROJECT_ROOT / "backend"

if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

load_dotenv(PROJECT_ROOT / ".env", override=True)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError(
        "QDRANT_URL and QDRANT_API_KEY must be set in the project .env."
    )

# Import application modules after loading .env.

COLLECTION_NAME = "research_documents"
BATCH_SIZE = 32
EXPECTED_VECTOR_SIZE = 768


def main() -> None:
    """Migrate persisted application chunks into Qdrant Cloud."""

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Configured endpoint: {QDRANT_URL}")

    module_url, module_api_key, _ = (
        qdrant_module._load_qdrant_settings()
    )
    print(f"Client factory endpoint: {module_url}")

    if module_url != QDRANT_URL:
        raise RuntimeError(
            "Qdrant endpoint configuration mismatch. "
            "Migration stopped without uploading data."
        )

    if not module_api_key:
        raise RuntimeError(
            "Qdrant API key is unavailable to the client factory."
        )

    with SessionLocal() as session:
        chunks = ChunkRepository(session).get_all_chunks()

    if not chunks:
        print("No persisted chunks found. Nothing to migrate.")
        return

    print(f"Loaded {len(chunks)} chunks from the application database.")

    client = qdrant_module.create_qdrant_client()

    try:
        collections = [
            item.name for item in client.get_collections().collections
        ]
        print(f"Collections visible to this client: {collections}")

        if COLLECTION_NAME not in collections:
            raise RuntimeError(
                f"Collection {COLLECTION_NAME!r} does not exist. "
                "Migration stopped without uploading data."
            )

        info = client.get_collection(COLLECTION_NAME)
        initial_count = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count
        repeated_count = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count
        metadata_count = client.get_collection(
            COLLECTION_NAME
        ).points_count

        print("Collection status:", info.status)
        print("Metadata count:", metadata_count)
        print("Initial exact count:", initial_count)
        print("Repeated exact count:", repeated_count)

        if not (
            initial_count == repeated_count == metadata_count
        ):
            raise RuntimeError(
                "Qdrant count discrepancy detected. "
                "No upload was attempted."
            )

        if initial_count != 0:
            raise RuntimeError(
                f"Migration stopped: {COLLECTION_NAME!r} contains "
                f"{initial_count} vectors. No upload was attempted."
            )

        embedder = BGEEmbedder()
        uploaded_count = 0

        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start: start + BATCH_SIZE]
            embeddings = embedder.embed_texts(
                [chunk["text"] for chunk in batch]
            )

            if len(embeddings) != len(batch):
                raise RuntimeError(
                    "Embedding count does not match chunk count."
                )

            points = []

            for chunk, embedding in zip(batch, embeddings):
                if len(embedding) != EXPECTED_VECTOR_SIZE:
                    raise RuntimeError(
                        f"Expected {EXPECTED_VECTOR_SIZE}-dimensional "
                        f"embedding; received {len(embedding)}."
                    )

                point = qdrant_module._create_point(chunk, embedding)

                if not isinstance(point, PointStruct):
                    raise TypeError("Expected a Qdrant PointStruct.")

                points.append(point)

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )

            uploaded_count += len(batch)
            print(f"Uploaded {uploaded_count}/{len(chunks)} chunks.")

        final_count = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count

        print(f"Expected vectors: {len(chunks)}")
        print(f"Actual vectors: {final_count}")

        if final_count != len(chunks):
            raise RuntimeError(
                f"Migration count mismatch: expected {len(chunks)}, "
                f"found {final_count}."
            )

        print("Migration completed and verified successfully.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
