from pydantic import BaseModel, Field, field_validator


class IngestionDocumentRequest(BaseModel):
    document_id: str = Field(
        min_length=1,
        description="Canonical document ID.",
    )

    document_name: str = Field(
        min_length=1,
        description="Original document filename.",
    )


class CreateIngestionJobRequest(BaseModel):
    documents: list[IngestionDocumentRequest] = Field(
        min_length=1,
        description="Documents to ingest.",
    )


class RegisteredDocumentsIngestionRequest(BaseModel):
    """
    Request model for creating an ingestion job
    from documents that are already registered.
    """

    document_ids: list[str] = Field(
        min_length=1,
        description="Canonical IDs of registered documents.",
    )

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(
        cls,
        document_ids: list[str],
    ) -> list[str]:
        if any(
            not document_id.strip()
            for document_id in document_ids
        ):
            raise ValueError(
                "Document IDs cannot be empty."
            )

        return document_ids


class CreateIngestionJobResponse(BaseModel):
    job_id: str
    status: str
    total_documents: int
