import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving meaningful content.
    """

    # Fix words split across lines because of PDF line wrapping.
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Replace remaining newlines with spaces.
    text = re.sub(r"\s*\n\s*", " ", text)

    # Collapse multiple whitespace characters into a single space.
    text = re.sub(r"\s+", " ", text)

    return text.strip()
