from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.database.document_repository import (
    DocumentRepository,
)
from app.schemas.documents import (
    RegisterDocumentRequest,
    RegisterDocumentResponse,
)
from app.services.document_registration_service import (
    DocumentRegistrationService,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


def get_document_registration_service(
    db: Session = Depends(get_db),
) -> DocumentRegistrationService:
    repository = DocumentRepository(
        session=db
    )

    return DocumentRegistrationService(
        repository=repository
    )


@router.post(
    "/register",
    response_model=RegisterDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_document(
    request: RegisterDocumentRequest,
    service: DocumentRegistrationService = Depends(
        get_document_registration_service
    ),
) -> RegisterDocumentResponse:
    try:
        document = service.register_document(
            file_path=request.file_path
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    return RegisterDocumentResponse(
        document_id=document.document_id,
        document_name=document.document_name,
        file_type=document.file_type,
        page_count=document.page_count,
        status=document.status,
    )


@router.post(
    "/upload",
    response_model=RegisterDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentRegistrationService = Depends(
        get_document_registration_service
    ),
) -> RegisterDocumentResponse:
    """
    Register a PDF uploaded through multipart/form-data.

    The PDF bytes are persisted in the database so that
    a separate worker process can access the document.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    try:
        file_content = await file.read()

        document = (
            service.register_uploaded_document(
                file_name=file.filename,
                file_content=file_content,
            )
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    finally:
        await file.close()

    return RegisterDocumentResponse(
        document_id=document.document_id,
        document_name=document.document_name,
        file_type=document.file_type,
        page_count=document.page_count,
        status=document.status,
    )
