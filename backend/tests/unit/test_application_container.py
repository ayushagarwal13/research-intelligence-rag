from unittest.mock import Mock

from app.services.application_container import ApplicationContainer


def test_application_container_refreshes_keyword_index_from_database(
    monkeypatch,
):
    persisted_chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "document-1",
            "document_name": "research.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "regularization improves inverse problems",
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "document-1",
            "document_name": "research.pdf",
            "page_number": 2,
            "chunk_index": 1,
            "text": "cross validation selects model parameters",
        },
    ]

    repository = Mock()
    repository.get_all_chunks.return_value = persisted_chunks

    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=None)

    monkeypatch.setattr(
        "app.services.application_container.SessionLocal",
        lambda: session,
    )

    monkeypatch.setattr(
        "app.services.application_container.ChunkRepository",
        lambda session: repository,
    )

    container = object.__new__(ApplicationContainer)

    container.keyword_search = Mock()

    container.refresh_keyword_index()

    repository.get_all_chunks.assert_called_once_with()

    container.keyword_search.update_chunks.assert_called_once_with(
        persisted_chunks
    )
