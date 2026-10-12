from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Request schema for a research question.
    """

    query: str = Field(
        min_length=1,
        description="Research question to answer.",
    )

    conversation_history: list[dict] | None = Field(
        default=None,
        description="Previous conversation messages.",
    )


class QuerySource(BaseModel):
    """
    Source information returned with an answer.
    """

    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    rerank_score: float


class QueryResponse(BaseModel):
    """
    Response returned by the RAG query endpoint.
    """

    answer: str
    sources: list[QuerySource]
