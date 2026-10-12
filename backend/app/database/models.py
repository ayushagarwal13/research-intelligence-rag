from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from app.database.base import Base


def utc_now() -> datetime:
    """
    Return the current timezone-aware UTC datetime.
    """
    return datetime.now(timezone.utc)


class DocumentModel(Base):
    """
    Canonical representation of a document in the
    research corpus.

    document_id is the SHA-256 hash of the document
    contents and therefore acts as the deduplication key.

    Documents may originate from either:
    - a local filesystem path, or
    - uploaded PDF bytes stored in the database.
    """

    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    document_name: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    file_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    document_path: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
    )

    document_content: Mapped[bytes | None] = mapped_column(
        LargeBinary,
        nullable=True,
    )

    page_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="registered",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    ingestion_documents: Mapped[
        list["IngestionDocumentModel"]
    ] = relationship(
        back_populates="document",
    )

    chunks: Mapped[
        list["DocumentChunkModel"]
    ] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class IngestionJobModel(Base):
    """
    Persistent database representation of an ingestion job.
    """

    __tablename__ = "ingestion_jobs"

    job_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )

    total_documents: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    completed_documents: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    failed_documents: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="queued",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    documents: Mapped[
        list["IngestionDocumentModel"]
    ] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )


class IngestionDocumentModel(Base):
    """
    Represents one document being processed as part
    of one ingestion job.

    A canonical document can participate in multiple
    ingestion jobs. Therefore the identity of this
    record is the combination of job_id and document_id.
    """

    __tablename__ = "ingestion_documents"

    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "ingestion_jobs.job_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )

    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "documents.document_id",
        ),
        primary_key=True,
    )

    document_name: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="queued",
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    job: Mapped[IngestionJobModel] = relationship(
        back_populates="documents",
    )

    document: Mapped[DocumentModel] = relationship(
        back_populates="ingestion_documents",
    )


class DocumentChunkModel(Base):
    """
    Persistent representation of one chunk produced
    from a research document.

    The chunk text and metadata are stored in PostgreSQL
    so the keyword/BM25 index can be rebuilt after an
    application restart.
    """

    __tablename__ = "document_chunks"

    chunk_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "documents.document_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    document_name: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    document: Mapped[DocumentModel] = relationship(
        back_populates="chunks",
    )
