"""
test_ocr.py — Tests para el módulo de extracción OCR.
"""

import pytest


def test_pdf_to_text_raises_not_implemented():
    """pdf_to_text debe lanzar NotImplementedError hasta que se implemente."""
    from pathlib import Path
    from src.pipeline.ocr_extract import pdf_to_text

    with pytest.raises(NotImplementedError):
        pdf_to_text(Path("fake.pdf"))


def test_extract_all_creates_output_dir(tmp_path):
    """extract_all debe crear el directorio de salida si no existe."""
    from src.pipeline.ocr_extract import extract_all

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    # Sin PDFs, no debería fallar
    count = extract_all(input_dir=input_dir, output_dir=output_dir)
    assert count == 0
    assert output_dir.exists()
