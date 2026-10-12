from app.schemas.ingestion_status import (
    IngestionJobStatusResponse,
)


def test_ingestion_job_status_response():
    response = IngestionJobStatusResponse(
        job_id="job-001",
        status="processing",
        total_documents=10,
        completed_documents=7,
        failed_documents=1,
    )

    assert response.job_id == "job-001"
    assert response.status == "processing"
    assert response.total_documents == 10
    assert response.completed_documents == 7
    assert response.failed_documents == 1
