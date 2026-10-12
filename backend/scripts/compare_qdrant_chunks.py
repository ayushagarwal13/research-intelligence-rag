from app.database.session import SessionLocal
from app.database.qdrant_client import create_qdrant_client
from app.database.chunk_repository import ChunkRepository
from dotenv import load_dotenv
from uuid import NAMESPACE_URL, uuid5
from pathlib import Path
import sys



PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
load_dotenv(PROJECT_ROOT / ".env", override=True)


COLLECTION_NAME = "research_documents"


def main() -> None:
    with SessionLocal() as session:
        chunks = ChunkRepository(session).get_all_chunks()

    local_ids = {chunk["chunk_id"] for chunk in chunks}
    expected_point_ids = {
        str(uuid5(NAMESPACE_URL, chunk_id))
        for chunk_id in local_ids
    }

    client = create_qdrant_client()

    try:
        cloud_point_ids = set()
        cloud_chunk_ids = set()
        offset = None

        while True:
            points, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                cloud_point_ids.add(str(point.id))

                if point.payload and point.payload.get("chunk_id"):
                    cloud_chunk_ids.add(point.payload["chunk_id"])

            if next_offset is None:
                break

            offset = next_offset

        print("SQLite chunks:", len(local_ids))
        print("Cloud points:", len(cloud_point_ids))
        print("Cloud payload chunk IDs:", len(cloud_chunk_ids))
        print("Matching expected point IDs:", len(
            expected_point_ids & cloud_point_ids))
        print("Local chunks missing from cloud:",
              len(local_ids - cloud_chunk_ids))
        print("Cloud chunk IDs not in SQLite:",
              len(cloud_chunk_ids - local_ids))

    finally:
        client.close()


if __name__ == "__main__":
    main()
