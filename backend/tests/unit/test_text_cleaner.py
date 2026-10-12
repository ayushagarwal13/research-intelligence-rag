import pytest

from app.ingestion.text_cleaner import clean_text


def test_clean_text_removes_line_breaks():
    text = "This is a line\nbroken into multiple lines."

    result = clean_text(text)

    assert result == "This is a line broken into multiple lines."


def test_clean_text_joins_hyphenated_words():
    text = "This is an impor-\ntant document."

    result = clean_text(text)

    assert result == "This is an important document."


def test_clean_text_normalizes_whitespace():
    text = "This   contains    extra\n\n whitespace."

    result = clean_text(text)

    assert result == "This contains extra whitespace."


def test_clean_text_strips_leading_and_trailing_spaces():
    text = "   This is a document.   "

    result = clean_text(text)

    assert result == "This is a document."


def test_clean_text_handles_empty_text():
    text = ""

    result = clean_text(text)

    assert result == ""
 