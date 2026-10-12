from unittest.mock import patch

from app.services.worker_dependencies import (
    create_ingestion_worker,
)


@patch(
    "app.services.worker_dependencies.BGEEmbedder"
)
@patch(
    "app.services.worker_dependencies.create_collection"
)
@patch(
    "app.services.worker_dependencies.create_qdrant_client"
)
def test_create_ingestion_worker(
    mock_create_qdrant_client,
    mock_create_collection,
    mock_bge_embedder,
):
    """
    Verify that the worker factory creates the
    required runtime dependencies.
    """

    mock_qdrant_client = (
        mock_create_qdrant_client.return_value
    )

    mock_embedder = mock_bge_embedder.return_value

    worker = create_ingestion_worker()

    assert worker.qdrant_client is mock_qdrant_client
    assert worker.embedder is mock_embedder

    assert worker.collection_name == (
        "research_documents"
    )

    assert worker.document_directory.name == (
        "documents"
    )

    assert worker.chunk_size == 500
    assert worker.overlap == 50

    assert worker.chunk_repository is None

    mock_create_collection.assert_called_once_with(
        client=mock_qdrant_client,
        collection_name="research_documents",
    )
