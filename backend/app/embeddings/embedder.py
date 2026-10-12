from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    """Generate embeddings using BAAI BGE."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
    ):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""

        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError(
                "Texts cannot contain empty strings."
            )

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()
