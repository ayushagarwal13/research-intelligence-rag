import hashlib
import re


def generate_chunk_id(
    document_id: str,
    page_number: int,
    chunk_index: int,
    text: str,
) -> str:
    """
    Generate a deterministic ID for a chunk
    using SHA-256 hashing.
    """

    unique_content = (
        f"{document_id}|"
        f"{page_number}|"
        f"{chunk_index}|"
        f"{text}"
    )

    return hashlib.sha256(
        unique_content.encode("utf-8")
    ).hexdigest()


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences while preserving
    sentence content.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_text(
    text: str,
    document_id: str,
    document_name: str,
    page_number: int,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    Split text into chunks while attempting to preserve
    sentence boundaries and document metadata.
    """

    if not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    sentences = _split_into_sentences(text)

    chunks = []
    current_text = ""
    chunk_index = 0

    for sentence in sentences:

        # Handle sentences longer than chunk_size.
        if len(sentence) > chunk_size:

            if current_text:
                chunks.append(
                    {
                        "chunk_id": generate_chunk_id(
                            document_id=document_id,
                            page_number=page_number,
                            chunk_index=chunk_index,
                            text=current_text,
                        ),
                        "document_id": document_id,
                        "document_name": document_name,
                        "page_number": page_number,
                        "text": current_text.strip(),
                        "chunk_index": chunk_index,
                    }
                )

                chunk_index += 1
                current_text = ""

            start = 0

            while start < len(sentence):
                end = start + chunk_size

                chunk_content = sentence[start:end].strip()

                if chunk_content:
                    chunks.append(
                        {
                            "chunk_id": generate_chunk_id(
                                document_id=document_id,
                                page_number=page_number,
                                chunk_index=chunk_index,
                                text=chunk_content,
                            ),
                            "document_id": document_id,
                            "document_name": document_name,
                            "page_number": page_number,
                            "text": chunk_content,
                            "chunk_index": chunk_index,
                        }
                    )

                    chunk_index += 1

                start += chunk_size - overlap

            continue

        proposed_text = (
            f"{current_text} {sentence}".strip()
            if current_text
            else sentence
        )

        if len(proposed_text) <= chunk_size:
            current_text = proposed_text

        else:

            if current_text:
                chunks.append(
                    {
                        "chunk_id": generate_chunk_id(
                            document_id=document_id,
                            page_number=page_number,
                            chunk_index=chunk_index,
                            text=current_text,
                        ),
                        "document_id": document_id,
                        "document_name": document_name,
                        "page_number": page_number,
                        "text": current_text,
                        "chunk_index": chunk_index,
                    }
                )

                chunk_index += 1

            # Preserve the last part of the previous chunk.
            overlap_text = (
                current_text[-overlap:]
                if current_text
                else ""
            )

            current_text = (
                f"{overlap_text} {sentence}".strip()
            )

            # Prevent overlap from exceeding the limit.
            if len(current_text) > chunk_size:
                current_text = sentence[:chunk_size].strip()

    # Add the final remaining chunk.
    if current_text:
        chunks.append(
            {
                "chunk_id": generate_chunk_id(
                    document_id=document_id,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    text=current_text,
                ),
                "document_id": document_id,
                "document_name": document_name,
                "page_number": page_number,
                "text": current_text.strip(),
                "chunk_index": chunk_index,
            }
        )

    return chunks
