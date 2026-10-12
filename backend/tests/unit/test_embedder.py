from app.embeddings.embedder import BGEEmbedder


def test_single_text_embedding():
    embedder = BGEEmbedder()

    embedding = embedder.embed_text(
        "Machine learning models learn from data."
    )

    assert isinstance(embedding, list)
    assert len(embedding) == 768


def test_batch_text_embeddings():
    embedder = BGEEmbedder()

    texts = [
        "Machine learning models learn from data.",
        "Deep learning uses neural networks.",
    ]

    embeddings = embedder.embed_texts(texts)

    assert len(embeddings) == 2

    for embedding in embeddings:
        assert isinstance(embedding, list)
        assert len(embedding) == 768


def test_empty_text_raises_error():
    embedder = BGEEmbedder()

    try:
        embedder.embed_text("")
        assert False, "Expected ValueError"
    except ValueError:
        pass
