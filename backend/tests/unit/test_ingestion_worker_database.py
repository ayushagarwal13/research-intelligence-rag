from dataclasses import dataclass
from pathlib import Path

import app.services.ingestion_worker as ingestion_worker_module
from app.services.ingestion_worker import IngestionWorker


@dataclass
class FakeDocument:
    document_id: str
    document_name: str
    status: str = "queued"
    error_message: str | None = None
    attempt_count: int = 0


class FakeRepository:
    def __init__(self, documents):
        self.documents = documents
        self.update_calls = []
        self.progress_calls = []
        self.commit_calls = 0

    def claim_next_queued_document(self, job_id):
        for document in self.documents:
            if document.status == "queued":
                document.status = "processing"
                document.attempt_count += 1
                document.error_message = None
                return document

        return None

    def update_document_model(self, document):
        self.update_calls.append(document)

    def update_job_progress(self, job_id):
        self.progress_calls.append(job_id)

    def commit(self):
        self.commit_calls += 1


def create_worker(tmp_path):
    return IngestionWorker(
        ingestion_job_service=None,
        qdrant_client=None,
        embedder=None,
        collection_name="test_collection",
        document_directory=str(tmp_path),
    )


def test_process_next_queued_document_returns_false_when_queue_is_empty(
    tmp_path,
):
    repository = FakeRepository([])

    worker = create_worker(tmp_path)

    result = worker.process_next_queued_document(
        repository=repository,
        job_id="job-1",
    )

    assert result is False
    assert repository.update_calls == []
    assert repository.progress_calls == []
    assert repository.commit_calls == 0


def test_process_next_queued_document_processes_claimed_document(
    tmp_path,
    monkeypatch,
):
    document = FakeDocument(
        document_id="document-1",
        document_name="paper.pdf",
    )

    pdf_path = Path(tmp_path) / "paper.pdf"
    pdf_path.write_bytes(b"fake pdf")

    repository = FakeRepository([document])

    worker = create_worker(tmp_path)

    monkeypatch.setattr(
        ingestion_worker_module,
        "extract_text_from_pdf",
        lambda path: {
            "document_id": "document-1",
            "document_name": "paper.pdf",
            "file_type": ".pdf",
            "document_path": str(path),
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "text": "Research content.",
                }
            ],
        },
    )

    monkeypatch.setattr(
        ingestion_worker_module,
        "index_document",
        lambda **kwargs: 1,
    )

    result = worker.process_next_queued_document(
        repository=repository,
        job_id="job-1",
    )

    assert result is True
    assert document.status == "completed"
    assert document.error_message is None
    assert document.attempt_count == 1

    assert repository.update_calls == [document]
    assert repository.progress_calls == ["job-1"]
    assert repository.commit_calls == 1


def test_process_next_queued_document_marks_failure(
    tmp_path,
    monkeypatch,
):
    document = FakeDocument(
        document_id="document-1",
        document_name="paper.pdf",
    )

    pdf_path = Path(tmp_path) / "paper.pdf"
    pdf_path.write_bytes(b"fake pdf")

    repository = FakeRepository([document])

    worker = create_worker(tmp_path)

    monkeypatch.setattr(
        ingestion_worker_module,
        "extract_text_from_pdf",
        lambda path: {
            "document_id": "document-1",
            "document_name": "paper.pdf",
            "file_type": ".pdf",
            "document_path": str(path),
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "text": "Research content.",
                }
            ],
        },
    )

    def failing_index(**kwargs):
        raise RuntimeError("Indexing failed.")

    monkeypatch.setattr(
        ingestion_worker_module,
        "index_document",
        failing_index,
    )

    result = worker.process_next_queued_document(
        repository=repository,
        job_id="job-1",
    )

    assert result is True
    assert document.status == "failed"
    assert document.error_message == "Indexing failed."
    assert document.attempt_count == 1

    assert repository.update_calls == [document]
    assert repository.progress_calls == ["job-1"]
    assert repository.commit_calls == 1


def test_process_queued_documents_processes_all_documents(
    tmp_path,
    monkeypatch,
):
    documents = [
        FakeDocument(
            document_id="document-1",
            document_name="paper1.pdf",
        ),
        FakeDocument(
            document_id="document-2",
            document_name="paper2.pdf",
        ),
        FakeDocument(
            document_id="document-3",
            document_name="paper3.pdf",
        ),
    ]

    for document in documents:
        (
            Path(tmp_path)
            / document.document_name
        ).write_bytes(b"fake pdf")

    repository = FakeRepository(documents)

    worker = create_worker(tmp_path)

    def fake_extract(path):
        document_name = Path(path).name

        document_id = {
            "paper1.pdf": "document-1",
            "paper2.pdf": "document-2",
            "paper3.pdf": "document-3",
        }[document_name]

        return {
            "document_id": document_id,
            "document_name": document_name,
            "file_type": ".pdf",
            "document_path": str(path),
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "text": "Research content.",
                }
            ],
        }

    monkeypatch.setattr(
        ingestion_worker_module,
        "extract_text_from_pdf",
        fake_extract,
    )

    monkeypatch.setattr(
        ingestion_worker_module,
        "index_document",
        lambda **kwargs: 1,
    )

    processed_count = (
        worker.process_queued_documents(
            repository=repository,
            job_id="job-1",
        )
    )

    assert processed_count == 3

    assert all(
        document.status == "completed"
        for document in documents
    )

    assert all(
        document.attempt_count == 1
        for document in documents
    )

    assert len(repository.update_calls) == 3
    assert repository.progress_calls == [
        "job-1",
        "job-1",
        "job-1",
    ]
    assert repository.commit_calls == 3
