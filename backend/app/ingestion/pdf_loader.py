from pathlib import Path
from app.ingestion.document_identity import generate_document_id

import pymupdf

from app.ingestion.text_cleaner import clean_text


def extract_text_from_pdf(file_path: str) -> dict:
    """
    Extract and clean text from a PDF while preserving
    document and page metadata.
    """
    pdf_path = Path(file_path)

    document = pymupdf.open(pdf_path)

    document_id = generate_document_id(str(pdf_path))

    pages = []

    for page_number, page in enumerate(document, start=1):
        raw_text = page.get_text()

        cleaned_text = clean_text(raw_text)

        pages.append(
            {
                "page_number": page_number,
                "text": cleaned_text,
            }
        )

    document.close()

    return {
        "document_id": document_id,
        "document_name": pdf_path.name,
        "file_type": pdf_path.suffix.lower(),
        "document_path": str(pdf_path),
        "page_count": len(pages),
        "pages": pages,
    }
