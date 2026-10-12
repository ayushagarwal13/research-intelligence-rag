from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Rerank retrieved chunks using a cross-encoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank chunks according to their relevance to the query.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not chunks:
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        pairs = [
            [query, chunk["text"]]
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        scored_chunks = []

        for chunk, score in zip(chunks, scores):
            result = chunk.copy()

            result["rerank_score"] = float(score)

            scored_chunks.append(result)

        scored_chunks.sort(
            key=lambda chunk: chunk["rerank_score"],
            reverse=True,
        )

        return scored_chunks[:top_k]
