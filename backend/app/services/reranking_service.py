from app.reranking.reranker import CrossEncoderReranker


def rerank_results(
    reranker: CrossEncoderReranker,
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank hybrid retrieval candidates and
    return the most relevant results.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    return reranker.rerank(
        query=query,
        chunks=candidates,
        top_k=top_k,
    )
