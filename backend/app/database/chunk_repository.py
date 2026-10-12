from sqlalchemy import select

from app.database.models import DocumentChunkModel


class ChunkRepository:
    """
    Repository for persistent document chunks.

    PostgreSQL is the durable source of truth for chunk text
    and metadata. The BM25 keyword index can be rebuilt from
    these records after an application restart.
    """

    def __init__(self, session):
        self.session = session

    def create_chunks(
        self,
        chunks: list[dict],
    ) -> None:
        """
        Persist document chunks.

        Existing chunks with the same chunk_id are replaced.
        This makes re-indexing deterministic and idempotent.
        """

        if not chunks:
            return

        for chunk in chunks:
            existing_chunk = self.session.get(
                DocumentChunkModel,
                chunk["chunk_id"],
            )

            if existing_chunk is not None:
                existing_chunk.document_id = chunk["document_id"]
                existing_chunk.document_name = chunk["document_name"]
                existing_chunk.page_number = chunk["page_number"]
                existing_chunk.chunk_index = chunk["chunk_index"]
                existing_chunk.text = chunk["text"]
                continue

            chunk_model = DocumentChunkModel(
                chunk_id=chunk["chunk_id"],
                document_id=chunk["document_id"],
                document_name=chunk["document_name"],
                page_number=chunk["page_number"],
                chunk_index=chunk["chunk_index"],
                text=chunk["text"],
            )

            self.session.add(chunk_model)

    def delete_document_chunks(
        self,
        document_id: str,
    ) -> int:
        """
        Delete all chunks belonging to a document.

        Returns:
            Number of chunks deleted.
        """

        chunks = self.session.scalars(
            select(DocumentChunkModel).where(
                DocumentChunkModel.document_id == document_id
            )
        ).all()

        for chunk in chunks:
            self.session.delete(chunk)

        return len(chunks)

    def get_chunks(
        self,
        document_id: str | None = None,
    ) -> list[dict]:
        """
        Retrieve persisted chunks.

        If document_id is provided, only chunks belonging
        to that document are returned.
        """

        statement = select(DocumentChunkModel)

        if document_id is not None:
            statement = statement.where(
                DocumentChunkModel.document_id == document_id
            )

        statement = statement.order_by(
            DocumentChunkModel.document_id,
            DocumentChunkModel.page_number,
            DocumentChunkModel.chunk_index,
        )

        models = self.session.scalars(statement).all()

        return [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
            }
            for chunk in models
        ]

    def get_all_chunks(self) -> list[dict]:
        """
        Retrieve all persisted chunks.

        This is primarily used to rebuild the BM25 keyword
        index when the application starts.
        """

        return self.get_chunks()

    def commit(self) -> None:
        """
        Commit the current transaction.
        """

        self.session.commit()
