import hashlib
from pathlib import Path


def generate_document_id(file_path: str) -> str:
    """
    Generate a deterministic document ID using
    the SHA-256 hash of the file contents.
    """
    pdf_path = Path(file_path)

    sha256_hash = hashlib.sha256()

    with pdf_path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()
