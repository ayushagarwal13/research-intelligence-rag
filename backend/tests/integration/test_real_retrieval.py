from pathlib import Path

from app.database.qdrant_client import (
    create_collection,
    create_qdrant_client,
)
from app.embeddings.embedder import BGEEmbedder
from app.ingestion.pdf_loader import extract_text_from_pdf
from app.services.document_indexing_service import index_document
from app.retrieval.vector_search import search_similar_chunks


def test_real_bge_retrieval():
    pdf_files = list(Path("data/documents").glob("*.pdf"))

    assert pdf_files, "No PDF files found."

    # Extract the first research paper.
    document = extract_text_from_pdf(str(pdf_files[0]))

    client = create_qdrant_client()
    collection_name = "real_retrieval_documents"

    create_collection(client, collection_name)

    # Load the real BGE embedding model.
    embedder = BGEEmbedder()

    # Index the document.
    indexed_count = index_document(
        client=client,
        embedder=embedder,
        collection_name=collection_name,
        document=document,
        chunk_size=500,
        overlap=50,
    )

    assert indexed_count > 0

    # Create an embedding for the research question.
    query = "What is the main topic of this research paper?"

    query_embedding = embedder.embed_text(query)

    # Retrieve the most similar chunks.
    results = search_similar_chunks(
        client=client,
        collection_name=collection_name,
        query_embedding=query_embedding,
        top_k=5,
    )

    assert len(results) > 0
    assert len(results) <= 5

    for result in results:
        assert result.payload is not None
        assert "text" in result.payload
        assert "document_name" in result.payload
        assert "page_number" in result.payload
