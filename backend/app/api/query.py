from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_query_service
from app.schemas.query import (
    QueryRequest,
    QueryResponse,
)
from app.services.llm_service import (
    LLMServiceUnavailableError,
)
from app.services.query_service import QueryService


router = APIRouter(
    prefix="/query",
    tags=["Query"],
)


@router.post(
    "",
    response_model=QueryResponse,
)
def query(
    request: QueryRequest,
    query_service: QueryService = Depends(
        get_query_service
    ),
) -> QueryResponse:
    """
    Answer a research question using the RAG pipeline.
    """

    try:
        result = query_service.answer_query(
            query=request.query,
            conversation_history=(
                request.conversation_history
            ),
        )

        return QueryResponse.model_validate(
            result
        )

    except LLMServiceUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
