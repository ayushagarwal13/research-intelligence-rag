from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    Range,
)


def search_similar_chunks(
    client: QdrantClient,
    collection_name: str,
    query_embedding: list[float],
    top_k: int = 5,
    document_id: str | None = None,
    page_number: int | None = None,
    min_page: int | None = None,
    max_page: int | None = None,
) -> list:
    """
    Retrieve similar chunks from Qdrant with optional
    metadata filtering.
    """

    if not query_embedding:
        raise ValueError(
            "Query embedding cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if page_number is not None and page_number <= 0:
        raise ValueError(
            "page_number must be greater than zero."
        )

    if min_page is not None and min_page <= 0:
        raise ValueError(
            "min_page must be greater than zero."
        )

    if max_page is not None and max_page <= 0:
        raise ValueError(
            "max_page must be greater than zero."
        )

    if (
        min_page is not None
        and max_page is not None
        and min_page > max_page
    ):
        raise ValueError(
            "min_page cannot be greater than max_page."
        )

    conditions = []

    if document_id is not None:
        conditions.append(
            FieldCondition(
                key="document_id",
                match=MatchValue(value=document_id),
            )
        )

    if page_number is not None:
        conditions.append(
            FieldCondition(
                key="page_number",
                match=MatchValue(value=page_number),
            )
        )

    if min_page is not None or max_page is not None:
        conditions.append(
            FieldCondition(
                key="page_number",
                range=Range(
                    gte=min_page,
                    lte=max_page,
                ),
            )
        )

    query_filter = (
        Filter(must=conditions)
        if conditions
        else None
    )

    search_results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    return search_results.points


def format_search_results(
    search_results: list,
) -> list[dict]:
    """
    Convert Qdrant search results into a clean structure
    for downstream RAG processing.
    """

    formatted_results = []

    for result in search_results:
        payload = result.payload or {}

        formatted_results.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "document_id": payload.get("document_id"),
                "document_name": payload.get("document_name"),
                "page_number": payload.get("page_number"),
                "chunk_index": payload.get("chunk_index"),
                "text": payload.get("text"),
                "similarity_score": result.score,
            }
        )

    return formatted_results
