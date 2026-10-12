from pydantic import BaseModel, Field


class RegisterDocumentRequest(BaseModel):
    """
    Request model for registering a physical document
    that already exists on the server filesystem.
    """

    file_path: str = Field(
        min_length=1,
        description="Path to the document file.",
    )


class RegisterDocumentResponse(BaseModel):
    """
    Response returned after document registration.
    """

    document_id: str
    document_name: str
    file_type: str
    page_count: int
    status: str
