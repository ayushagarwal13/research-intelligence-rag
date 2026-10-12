from pathlib import Path

import pymupdf
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database.document_repository import (
    DocumentRepository,
)
from app.database.ingestion_repository import (
    IngestionRepository,
)
from app.database.models import (
    DocumentModel,
    IngestionDocumentModel,
    IngestionJobModel,
)
from app.services.document_registration_service import (
    DocumentRegistrationService,
)
from app.services.ingestion_submission_service import (
    IngestionSubmissionService,
)


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


def create_test_pdf(
    file_path: Path,
) -> None:
    """
    Create a minimal valid PDF for integration testing.
    """
    document = pymupdf.open()

    page = document.new_page()

    page.insert_text(
        (72, 72),
        "This is a research document "
        "used for ingestion submission testing.",
    )

    document.save(file_path)
    document.close()


def test_registered_documents_can_be_submitted_for_ingestion(
    tmp_path: Path,
):
    setup_test_database()

    try:
        document_path = (
            tmp_path / "paper.pdf"
        )

        create_test_pdf(
            document_path
        )

        with TestSessionLocal() as db:
            document_repository = (
                DocumentRepository(
                    session=db
                )
            )

            ingestion_repository = (
                IngestionRepository(
                    session=db
                )
            )

            registration_service = (
                DocumentRegistrationService(
                    repository=document_repository
                )
            )

            registered_document = (
                registration_service.register_document(
                    str(document_path)
                )
            )

            submission_service = (
                IngestionSubmissionService(
                    document_repository=(
                        document_repository
                    ),
                    ingestion_repository=(
                        ingestion_repository
                    ),
                )
            )

            job, ingestion_documents = (
                submission_service.submit_documents(
                    file_paths=[
                        str(document_path)
                    ]
                )
            )

            assert job.total_documents == 1

            assert len(
                ingestion_documents
            ) == 1

            assert (
                ingestion_documents[0].document_id
                == registered_document.document_id
            )

            assert (
                ingestion_documents[0].document_name
                == registered_document.document_name
            )

            document = db.get(
                DocumentModel,
                registered_document.document_id,
            )

            assert document is not None

            assert (
                document.document_id
                == registered_document.document_id
            )

            assert (
                document.status
                == "registered"
            )

            stored_job = db.get(
                IngestionJobModel,
                job.job_id,
            )

            assert stored_job is not None

            assert (
                stored_job.total_documents
                == 1
            )

            assert (
                stored_job.status
                == "queued"
            )

            statement = select(
                IngestionDocumentModel
            ).where(
                IngestionDocumentModel.job_id
                == job.job_id
            )

            stored_ingestion_document = (
                db.scalars(
                    statement
                ).one()
            )

            assert (
                stored_ingestion_document.document_id
                == registered_document.document_id
            )

            assert (
                stored_ingestion_document.status
                == "queued"
            )

            assert (
                stored_ingestion_document.attempt_count
                == 0
            )

    finally:
        teardown_test_database()
