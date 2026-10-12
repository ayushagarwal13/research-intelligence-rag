
from rank_bm25 import BM25Okapi


class KeywordSearch:
    """Keyword-based retrieval using BM25."""

    def __init__(self, chunks: list[dict]):
        if not chunks:
            raise ValueError("Chunks cannot be empty.")

        self.chunks: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self.update_chunks(chunks)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def update_chunks(self, chunks: list[dict]) -> None:
        """Replace the indexed chunks and rebuild the BM25 index."""
        self.chunks = chunks or []

        tokenized_corpus = [
            self._tokenize(chunk.get("text", ""))
            for chunk in self.chunks
        ]

        if not tokenized_corpus or not any(tokenized_corpus):
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return up to top_k chunks ranked by BM25 score."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.bm25 is None or not self.chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:top_k]:
            result = self.chunks[index].copy()
            result["keyword_score"] = float(scores[index])
            results.append(result)

        return results
