def reciprocal_rank_fusion(
    retrieval_results: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Combine ranked results from multiple retrieval systems
    using Reciprocal Rank Fusion.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    scores = {}
    chunks = {}

    for results in retrieval_results:
        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk["chunk_id"]

            score = 1 / (k + rank)

            scores[chunk_id] = (
                scores.get(chunk_id, 0) + score
            )

            chunks[chunk_id] = chunk.copy()

    ranked_chunk_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )

    fused_results = []

    for chunk_id in ranked_chunk_ids:
        chunk = chunks[chunk_id]

        chunk["rrf_score"] = scores[chunk_id]

        fused_results.append(chunk)

    return fused_results
