from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.base import Base
from app.database.chunk_repository import ChunkRepository
from app.database.models import DocumentModel


def create_test_session():
    """
    Create a fresh in-memory SQLite database for testing.
    """

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={
            "check_same_thread": False,
        },
    )

    Base.metadata.create_all(engine)

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    return session_factory()


def create_document(
    session,
    document_id: str = "a" * 64,
):
    """
    Create a canonical document required by the
    document_chunks foreign key.
    """

    document = DocumentModel(
        document_id=document_id,
        document_name="research-paper.pdf",
        file_type=".pdf",
        document_path="data/documents/research-paper.pdf",
        page_count=2,
        status="registered",
    )

    session.add(document)
    session.commit()

    return document


def sample_chunks(
    document_id: str = "a" * 64,
    chunk_prefix: str = "chunk",
) -> list[dict]:
    """
    Return representative document chunks.

    chunk IDs are globally unique because chunk_id is the
    primary key of the document_chunks table.
    """

    return [
        {
            "chunk_id": f"{chunk_prefix}-001",
            "document_id": document_id,
            "document_name": "research-paper.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": (
                "Retrieval augmented generation combines "
                "retrieval with generation."
            ),
        },
        {
            "chunk_id": f"{chunk_prefix}-002",
            "document_id": document_id,
            "document_name": "research-paper.pdf",
            "page_number": 2,
            "chunk_index": 1,
            "text": (
                "Hybrid retrieval combines dense and "
                "keyword based search."
            ),
        },
    ]


def test_create_and_get_chunks():
    session = create_test_session()

    create_document(session)

    repository = ChunkRepository(session)

    chunks = sample_chunks()

    repository.create_chunks(chunks)
    repository.commit()

    stored_chunks = repository.get_all_chunks()

    assert len(stored_chunks) == 2
    assert stored_chunks[0]["chunk_id"] == "chunk-001"
    assert stored_chunks[0]["page_number"] == 1
    assert stored_chunks[1]["chunk_id"] == "chunk-002"
    assert stored_chunks[1]["page_number"] == 2

    session.close()


def test_get_chunks_by_document_id():
    session = create_test_session()

    document_id = "a" * 64
    create_document(
        session,
        document_id=document_id,
    )

    second_document_id = "b" * 64
    create_document(
        session,
        document_id=second_document_id,
    )

    repository = ChunkRepository(session)

    first_document_chunks = sample_chunks(
        document_id=document_id,
        chunk_prefix="document-a-chunk",
    )

    second_document_chunks = sample_chunks(
        document_id=second_document_id,
        chunk_prefix="document-b-chunk",
    )

    repository.create_chunks(
        first_document_chunks
        + second_document_chunks
    )
    repository.commit()

    first_document_results = repository.get_chunks(
        document_id=document_id,
    )

    assert len(first_document_results) == 2

    assert all(
        chunk["document_id"] == document_id
        for chunk in first_document_results
    )

    session.close()


def test_delete_document_chunks():
    session = create_test_session()

    document_id = "a" * 64

    create_document(
        session,
        document_id=document_id,
    )

    repository = ChunkRepository(session)

    repository.create_chunks(
        sample_chunks(document_id)
    )
    repository.commit()

    deleted_count = repository.delete_document_chunks(
        document_id
    )
    repository.commit()

    assert deleted_count == 2

    assert repository.get_chunks(
        document_id=document_id
    ) == []

    session.close()


def test_create_chunks_is_idempotent():
    session = create_test_session()

    document_id = "a" * 64

    create_document(
        session,
        document_id=document_id,
    )

    repository = ChunkRepository(session)

    chunks = sample_chunks(document_id)

    repository.create_chunks(chunks)
    repository.commit()

    updated_chunks = [
        {
            **chunks[0],
            "text": "Updated chunk content.",
        },
        chunks[1],
    ]

    repository.create_chunks(updated_chunks)
    repository.commit()

    stored_chunks = repository.get_all_chunks()

    assert len(stored_chunks) == 2

    assert stored_chunks[0]["text"] == (
        "Updated chunk content."
    )

    session.close()
