import re
from pathlib import Path

from app.ingestion.pdf_loader import extract_text_from_pdf


def test_all_pdfs_can_be_ingested():
    pdf_directory = Path("data/documents")

    pdf_files = sorted(pdf_directory.glob("*.pdf"))

    assert len(pdf_files) == 10, (
        f"Expected 10 PDFs, found {len(pdf_files)}"
    )

    for pdf_path in pdf_files:
        document = extract_text_from_pdf(str(pdf_path))

        assert document["document_name"] == pdf_path.name
        assert document["page_count"] > 0
        assert document["pages"]

        for page in document["pages"]:
            assert page["page_number"] > 0
            assert isinstance(page["text"], str)

        extracted_text = "\n".join(
            page["text"] for page in document["pages"]
        )

        assert extracted_text.strip(), (
            f"No text extracted from {pdf_path.name}"
        )

        print(
            f"\n✓ {pdf_path.name}"
            f" | Pages: {document['page_count']}"
            f" | Characters: {len(extracted_text)}"
        )


def test_pdf_pages_contain_cleaned_text():
    pdf_directory = Path("data/documents")
    pdf_path = sorted(pdf_directory.glob("*.pdf"))[0]

    document = extract_text_from_pdf(str(pdf_path))

    for page in document["pages"]:
        text = page["text"]

        # Verify that unnecessary line breaks were removed.
        assert "\n" not in text

        # Verify that consecutive whitespace was normalized.
        assert not re.search(r"\s{2,}", text)

        # Verify that page metadata is preserved.
        assert page["page_number"] > 0

        # Verify that text remains available.
        assert isinstance(text, str)
        assert text.strip()
