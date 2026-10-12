from datetime import timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database.document_repository import (
    DocumentRepository,
)
from app.database.models import DocumentModel


TEST_DATABASE_URL = "sqlite://"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={
        "check_same_thread": False,
    },
    poolclass=StaticPool,
)

TestSessionLocal = sessionmaker(
    bind=test_engine,
    autoflush=False,
    autocommit=False,
)


def setup_test_database() -> None:
    Base.metadata.create_all(
        bind=test_engine
    )


def teardown_test_database() -> None:
    Base.metadata.drop_all(
        bind=test_engine
    )


def create_document() -> DocumentModel:
    return DocumentModel(
        document_id="document-001",
        document_name="research-paper.pdf",
        file_type=".pdf",
        document_path="data/documents/research-paper.pdf",
        page_count=10,
        status="registered",
    )


def test_create_and_get_document():
    setup_test_database()

    try:
        with TestSessionLocal() as db:
            repository = DocumentRepository(
                session=db
            )

            document = create_document()

            repository.create_document(
                document
            )
            repository.commit()

            retrieved_document = (
                repository.get_document(
                    "document-001"
                )
            )

            assert retrieved_document is not None
            assert (
                retrieved_document.document_id
                == "document-001"
            )
            assert (
                retrieved_document.document_name
                == "research-paper.pdf"
            )
            assert (
                retrieved_document.file_type
                == ".pdf"
            )
            assert (
                retrieved_document.page_count
                == 10
            )
            assert (
                retrieved_document.status
                == "registered"
            )

    finally:
        teardown_test_database()


def test_document_exists():
    setup_test_database()

    try:
        with TestSessionLocal() as db:
            repository = DocumentRepository(
                session=db
            )

            document = create_document()

            repository.create_document(
                document
            )
            repository.commit()

            assert (
                repository.document_exists(
                    "document-001"
                )
                is True
            )

            assert (
                repository.document_exists(
                    "does-not-exist"
                )
                is False
            )

    finally:
        teardown_test_database()


def test_get_document_returns_none_for_missing_document():
    setup_test_database()

    try:
        with TestSessionLocal() as db:
            repository = DocumentRepository(
                session=db
            )

            document = repository.get_document(
                "does-not-exist"
            )

            assert document is None

    finally:
        teardown_test_database()


def test_update_document():
    setup_test_database()

    try:
        with TestSessionLocal() as db:
            repository = DocumentRepository(
                session=db
            )

            document = create_document()

            repository.create_document(
                document
            )
            repository.commit()

            document.document_name = (
                "updated-paper.pdf"
            )
            document.page_count = 15
            document.status = "completed"

            repository.update_document(
                document
            )
            repository.commit()

            updated_document = (
                repository.get_document(
                    "document-001"
                )
            )

            assert updated_document is not None
            assert (
                updated_document.document_name
                == "updated-paper.pdf"
            )
            assert (
                updated_document.page_count
                == 15
            )
            assert (
                updated_document.status
                == "completed"
            )

            assert (
                updated_document.updated_at
                is not None
            )

            if (
                updated_document.updated_at.tzinfo
                is not None
            ):
                assert (
                    updated_document.updated_at.tzinfo
                    == timezone.utc
                )

    finally:
        teardown_test_database()


def test_update_missing_document_raises_error():
    setup_test_database()

    try:
        with TestSessionLocal() as db:
            repository = DocumentRepository(
                session=db
            )

            document = create_document()

            try:
                repository.update_document(
                    document
                )
            except ValueError as error:
                assert (
                    str(error)
                    == "Document not found: document-001"
                )
            else:
                raise AssertionError(
                    "Expected ValueError"
                )

    finally:
        teardown_test_database()
