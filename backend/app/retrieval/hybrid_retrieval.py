from app.retrieval.keyword_search import KeywordSearch
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.vector_search import (
    format_search_results,
    search_similar_chunks,
)


def hybrid_search(
    client,
    embedder,
    collection_name: str,
    keyword_search: KeywordSearch,
    query: str,
    dense_top_k: int = 10,
    keyword_top_k: int = 10,
    rrf_top_k: int = 20,
) -> list[dict]:
    """
    Perform dense and keyword retrieval,
    combine results using RRF, and return
    candidates for reranking.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if dense_top_k <= 0:
        raise ValueError(
            "dense_top_k must be greater than zero."
        )

    if keyword_top_k <= 0:
        raise ValueError(
            "keyword_top_k must be greater than zero."
        )

    if rrf_top_k <= 0:
        raise ValueError(
            "rrf_top_k must be greater than zero."
        )

    # ---------------------------------
    # 1. Dense retrieval
    # ---------------------------------

    query_embedding = embedder.embed_text(query)

    dense_results = search_similar_chunks(
        client=client,
        collection_name=collection_name,
        query_embedding=query_embedding,
        top_k=dense_top_k,
    )

    dense_results = format_search_results(
        dense_results
    )

    # ---------------------------------
    # 2. BM25 keyword retrieval
    # ---------------------------------

    keyword_results = keyword_search.search(
        query=query,
        top_k=keyword_top_k,
    )

    # ---------------------------------
    # 3. Combine rankings using RRF
    # ---------------------------------

    fused_results = reciprocal_rank_fusion(
        retrieval_results=[
            dense_results,
            keyword_results,
        ]
    )

    # ---------------------------------
    # 4. Return RRF candidates
    # ---------------------------------

    return fused_results[:rrf_top_k]
