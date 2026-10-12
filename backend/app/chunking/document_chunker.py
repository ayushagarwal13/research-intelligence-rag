from app.chunking.chunker import chunk_text


def chunk_document(
    document: dict,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    Chunk all pages of an extracted document.

    Args:
        document: Extracted document containing metadata and pages.
        chunk_size: Maximum target size of each chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        A list of chunks containing text and metadata.
    """

    document_id = document["document_id"]
    document_name = document["document_name"]

    all_chunks = []

    for page in document["pages"]:
        page_number = page["page_number"]
        page_text = page["text"]

        page_chunks = chunk_text(
            text=page_text,
            document_id=document_id,
            document_name=document_name,
            page_number=page_number,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        all_chunks.extend(page_chunks)

    return all_chunks
