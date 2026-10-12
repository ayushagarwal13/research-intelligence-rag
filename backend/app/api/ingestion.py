from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.database.document_repository import (
    DocumentRepository,
)
from app.database.ingestion_repository import (
    IngestionRepository,
)
from app.schemas.ingestion import (
    CreateIngestionJobResponse,
    RegisteredDocumentsIngestionRequest,
)
from app.schemas.ingestion_status import (
    IngestionJobStatusResponse,
)
from app.services.ingestion_status_service import (
    IngestionStatusService,
)
from app.services.ingestion_submission_service import (
    IngestionSubmissionService,
)


router = APIRouter(
    prefix="/ingestion",
    tags=["Ingestion"],
)


def get_ingestion_submission_service(
    db: Session = Depends(get_db),
) -> IngestionSubmissionService:
    document_repository = DocumentRepository(
        session=db
    )

    ingestion_repository = IngestionRepository(
        session=db
    )

    return IngestionSubmissionService(
        document_repository=document_repository,
        ingestion_repository=ingestion_repository,
    )


def get_ingestion_status_service(
    db: Session = Depends(get_db),
) -> IngestionStatusService:
    repository = IngestionRepository(
        session=db
    )

    return IngestionStatusService(
        repository=repository
    )


@router.post(
    "/jobs",
    response_model=CreateIngestionJobResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_ingestion_job(
    request: RegisteredDocumentsIngestionRequest,
    service: IngestionSubmissionService = Depends(
        get_ingestion_submission_service
    ),
) -> CreateIngestionJobResponse:
    try:
        job, _ = (
            service.submit_registered_documents(
                document_ids=request.document_ids
            )
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    return CreateIngestionJobResponse(
        job_id=job.job_id,
        status=job.status,
        total_documents=job.total_documents,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=IngestionJobStatusResponse,
)
def get_ingestion_job_status(
    job_id: str,
    service: IngestionStatusService = Depends(
        get_ingestion_status_service
    ),
) -> IngestionJobStatusResponse:
    try:
        job = service.get_job_status(
            job_id=job_id
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    return IngestionJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        total_documents=job.total_documents,
        completed_documents=job.completed_documents,
        failed_documents=job.failed_documents,
    )
