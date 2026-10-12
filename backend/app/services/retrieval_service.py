from app.retrieval.hybrid_retrieval import hybrid_search
from app.retrieval.keyword_search import KeywordSearch
from app.retrieval.query_rewriter import QueryRewriter
from app.reranking.reranker import CrossEncoderReranker


def retrieve_relevant_chunks(
    client,
    embedder,
    reranker: CrossEncoderReranker,
    collection_name: str,
    keyword_search: KeywordSearch,
    query: str,
    query_rewriter: QueryRewriter | None = None,
    conversation_history: list[dict] | None = None,
    dense_top_k: int = 10,
    keyword_top_k: int = 10,
    rrf_top_k: int = 20,
    final_top_k: int = 5,
) -> list[dict]:
    """
    Perform the complete retrieval pipeline:

    1. Rewrite conversational query if required.
    2. Dense vector retrieval.
    3. BM25 keyword retrieval.
    4. Reciprocal Rank Fusion.
    5. Cross-encoder reranking.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if final_top_k <= 0:
        raise ValueError(
            "final_top_k must be greater than zero."
        )

    # ---------------------------------
    # 1. Query rewriting
    # ---------------------------------

    retrieval_query = query

    if query_rewriter is not None:
        retrieval_query = query_rewriter.rewrite(
            query=query,
            conversation_history=conversation_history,
        )

    # ---------------------------------
    # 2. Hybrid retrieval
    # ---------------------------------

    candidates = hybrid_search(
        client=client,
        embedder=embedder,
        collection_name=collection_name,
        keyword_search=keyword_search,
        query=retrieval_query,
        dense_top_k=dense_top_k,
        keyword_top_k=keyword_top_k,
        rrf_top_k=rrf_top_k,
    )

    # ---------------------------------
    # 3. Cross-encoder reranking
    # ---------------------------------

    final_results = reranker.rerank(
        query=retrieval_query,
        chunks=candidates,
        top_k=final_top_k,
    )

    return final_results
