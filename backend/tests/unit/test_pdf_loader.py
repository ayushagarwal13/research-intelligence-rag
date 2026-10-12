from pathlib import Path

from app.ingestion.pdf_loader import extract_text_from_pdf


def test_extract_text_from_pdf():
    pdf_directory = Path("data/documents")

    pdf_files = list(pdf_directory.glob("*.pdf"))

    assert pdf_files, "No PDF files found in data/documents"

    pdf_path = pdf_files[0]

    document = extract_text_from_pdf(str(pdf_path))

    assert document["document_id"]
    assert document["document_name"] == pdf_path.name
    assert document["file_type"] == ".pdf"
    assert document["document_path"] == str(pdf_path)
    assert document["page_count"] > 0
    assert document["pages"]

    for page in document["pages"]:
        assert page["page_number"] > 0
        assert isinstance(page["text"], str)

    extracted_text = "\n".join(
        page["text"] for page in document["pages"]
    )

    assert extracted_text.strip(), "No text was extracted from the PDF"

    print(f"\nPDF tested: {document['document_name']}")
    print(f"Number of pages: {document['page_count']}")
    print(f"Extracted characters: {len(extracted_text)}")

    print("\nFirst page text:\n")
    print(document["pages"][0]["text"][:1000])


def test_document_id_remains_consistent():
    pdf_directory = Path("data/documents")

    pdf_path = next(pdf_directory.glob("*.pdf"))

    first_document = extract_text_from_pdf(str(pdf_path))
    second_document = extract_text_from_pdf(str(pdf_path))

    assert first_document["document_id"] == second_document["document_id"]
