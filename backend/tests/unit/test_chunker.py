from app.chunking.chunker import chunk_text


def test_short_text_returns_one_chunk():
    text = "This is a short research paragraph."

    chunks = chunk_text(
        text=text,
        document_id="doc-1",
        document_name="research.pdf",
        page_number=1,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["document_id"] == "doc-1"
    assert chunks[0]["document_name"] == "research.pdf"
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["chunk_index"] == 0


def test_long_text_is_split_into_multiple_chunks():
    text = "A" * 250

    chunks = chunk_text(
        text=text,
        document_id="doc-1",
        document_name="research.pdf",
        page_number=1,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk["text"]) <= 100


def test_chunks_have_unique_indexes():
    text = "Research text. " * 30

    chunks = chunk_text(
        text=text,
        document_id="doc-1",
        document_name="research.pdf",
        page_number=2,
        chunk_size=100,
        overlap=20,
    )

    indexes = [chunk["chunk_index"] for chunk in chunks]

    assert indexes == list(range(len(chunks)))


def test_empty_text_returns_no_chunks():
    chunks = chunk_text(
        text="",
        document_id="doc-1",
        document_name="research.pdf",
        page_number=1,
        chunk_size=100,
        overlap=20,
    )

    assert chunks == []


def test_chunk_ids_are_unique():
    text = "Research text. " * 30

    chunks = chunk_text(
        text=text,
        document_id="doc-1",
        document_name="research.pdf",
        page_number=1,
        chunk_size=100,
        overlap=20,
    )

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunk_ids_are_reproducible():
    text = (
        "Machine learning models learn patterns from data. "
        "These patterns help models make predictions. "
        "Evaluation measures model performance."
    )

    first_chunks = chunk_text(
        text=text,
        document_id="doc-1",
        document_name="research.pdf",
        page_number=1,
        chunk_size=100,
        overlap=10,
    )

    second_chunks = chunk_text(
        text=text,
        document_id="doc-1",
        document_name="research.pdf",
        page_number=1,
        chunk_size=100,
        overlap=10,
    )

    first_ids = [chunk["chunk_id"] for chunk in first_chunks]
    second_ids = [chunk["chunk_id"] for chunk in second_chunks]

    assert first_ids == second_ids


def test_chunks_do_not_exceed_chunk_size():
    text = (
        "This is a sentence containing enough words "
        "to require multiple chunks. "
    ) * 20

    chunks = chunk_text(
        text=text,
        document_id="doc-1",
        document_name="test.pdf",
        page_number=1,
        chunk_size=100,
        overlap=20,
    )

    assert chunks

    for chunk in chunks:
        assert len(chunk["text"]) <= 100


def test_chunks_contain_overlap():
    text = (
        "Sentence one contains important information. "
        "Sentence two contains additional information. "
        "Sentence three contains more information. "
        "Sentence four contains final information."
    )

    chunks = chunk_text(
        text=text,
        document_id="doc-1",
        document_name="test.pdf",
        page_number=1,
        chunk_size=80,
        overlap=20,
    )

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:]):
        previous_end = previous["text"][-20:]

        assert previous_end in current["text"]



