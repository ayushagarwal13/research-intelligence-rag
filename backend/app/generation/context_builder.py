def build_context(
    chunks: list[dict],
) -> str:
    """
    Build a structured context string from
    retrieved chunks.

    Each chunk includes source metadata so that
    the generation model can identify where the
    information came from.
    """

    if not chunks:
        return ""

    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        document_name = chunk.get(
            "document_name",
            "Unknown document",
        )

        page_number = chunk.get(
            "page_number",
            "Unknown page",
        )

        text = chunk.get(
            "text",
            "",
        ).strip()

        if not text:
            continue

        context_parts.append(
            f"[Source {index}]\n"
            f"Document: {document_name}\n"
            f"Page: {page_number}\n"
            f"Content:\n{text}"
        )

    return "\n\n".join(context_parts)
