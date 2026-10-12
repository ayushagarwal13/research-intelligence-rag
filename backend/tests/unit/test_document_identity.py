from pathlib import Path

from app.ingestion.document_identity import generate_document_id


def test_document_id_is_deterministic(tmp_path: Path):
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"sample pdf content")

    first_id = generate_document_id(str(file_path))
    second_id = generate_document_id(str(file_path))

    assert first_id == second_id


def test_different_files_have_different_ids(tmp_path: Path):
    first_file = tmp_path / "first.pdf"
    second_file = tmp_path / "second.pdf"

    first_file.write_bytes(b"first content")
    second_file.write_bytes(b"second content")

    first_id = generate_document_id(str(first_file))
    second_id = generate_document_id(str(second_file))

    assert first_id != second_id


def test_document_id_is_sha256_hex(tmp_path: Path):
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"sample pdf content")

    document_id = generate_document_id(str(file_path))

    assert len(document_id) == 64
    assert all(character in "0123456789abcdef" for character in document_id)
