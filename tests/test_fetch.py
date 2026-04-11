"""
test_fetch.py — Tests para el módulo de consulta a OpenAlex.
"""

import pytest


def test_fetch_works_returns_list():
    """fetch_works debe retornar una lista de diccionarios."""
    # TODO: Implementar con mock de la API
    pass


def test_save_abstracts_creates_files(tmp_path):
    """save_abstracts debe crear archivos .txt en el directorio de salida."""
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
    """save_abstracts debe ignorar trabajos sin abstract."""
    from src.pipeline.fetch_openalex import save_abstracts

    works = [
        {"id": "https://openalex.org/W789", "abstract": ""},
        {"id": "https://openalex.org/W000", "abstract": "Valid abstract."},
    ]

    save_abstracts(works, output_dir=tmp_path)

    assert not (tmp_path / "W789.txt").exists()
    assert (tmp_path / "W000.txt").exists()
