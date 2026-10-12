from app.embeddings.embedder import BGEEmbedder
from app.embeddings.similarity import cosine_similarity


def test_semantic_similarity():
    embedder = BGEEmbedder()

    text_a = "Machine learning models learn from data."
    text_b = "Algorithms learn patterns from datasets."
    text_c = "Cooking requires ingredients and heat."

    embedding_a = embedder.embed_text(text_a)
    embedding_b = embedder.embed_text(text_b)
    embedding_c = embedder.embed_text(text_c)

    similarity_ab = cosine_similarity(embedding_a, embedding_b)
    similarity_ac = cosine_similarity(embedding_a, embedding_c)

    print(f"\nSimilarity A-B: {similarity_ab}")
    print(f"Similarity A-C: {similarity_ac}")

    assert similarity_ab > similarity_ac
