from pathlib import Path

from app.database.qdrant_client import (
    create_collection,
    create_qdrant_client,
)
from app.embeddings.embedder import BGEEmbedder
from app.ingestion.pdf_loader import extract_text_from_pdf
from app.services.document_indexing_service import index_document


def test_real_document_indexing():
    pdf_files = list(Path("data/documents").glob("*.pdf"))

    assert pdf_files, "No PDF files found."

    document = extract_text_from_pdf(str(pdf_files[0]))

    client = create_qdrant_client()
    collection_name = "real_research_documents"

    create_collection(client, collection_name)

    embedder = BGEEmbedder()

    indexed_count = index_document(
        client=client,
        embedder=embedder,
        collection_name=collection_name,
        document=document,
        chunk_size=500,
        overlap=50,
    )

    stored_count = client.count(
        collection_name=collection_name,
        exact=True,
    ).count

    assert indexed_count > 0
    assert stored_count == indexed_count
