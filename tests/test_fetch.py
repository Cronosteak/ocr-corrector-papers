"""
test_fetch.py — Tests for the OpenAlex query module.
"""

import pytest


def test_fetch_works_returns_list():
    """fetch_works should return a list of dicts."""
    # TODO: implement with a mocked API
    pass


def test_save_abstracts_creates_files(tmp_path):
    """save_abstracts should create .txt files in the output directory."""
    from src.pipeline.fetch_openalex import save_abstracts

    works = [
        {"id": "https://openalex.org/W123", "abstract": "This is a test abstract."},
        {"id": "https://openalex.org/W456", "abstract": "Another abstract here."},
    ]

    save_abstracts(works, output_dir=tmp_path)

    assert (tmp_path / "W123.txt").exists()
    assert (tmp_path / "W456.txt").exists()
    assert (tmp_path / "W123.txt").read_text() == "This is a test abstract."


def test_save_abstracts_skips_empty(tmp_path):
    """save_abstracts should skip works without an abstract."""
    from src.pipeline.fetch_openalex import save_abstracts

    works = [
        {"id": "https://openalex.org/W789", "abstract": ""},
        {"id": "https://openalex.org/W000", "abstract": "Valid abstract."},
    ]

    save_abstracts(works, output_dir=tmp_path)

    assert not (tmp_path / "W789.txt").exists()
    assert (tmp_path / "W000.txt").exists()
