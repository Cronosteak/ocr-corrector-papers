"""
test_ocr.py — Tests for the OCR extraction module.
"""

import pytest


def test_pdf_to_text_raises_on_missing_file():
    """pdf_to_text should propagate the error when the PDF does not exist."""
    from pathlib import Path
    from pdf2image.exceptions import PDFPageCountError
    from src.pipeline.ocr_extract import pdf_to_text

    with pytest.raises(PDFPageCountError):
        pdf_to_text(Path("fake.pdf"))


def test_pdf_to_text_joins_pages(monkeypatch):
    """Pages are concatenated separated by a blank line."""
    from pathlib import Path
    from src.pipeline import ocr_extract

    monkeypatch.setattr(ocr_extract, "convert_from_path", lambda *a, **k: ["img1", "img2"])
    monkeypatch.setattr(
        ocr_extract.pytesseract, "image_to_string", lambda img, lang="eng": f"text of {img}"
    )

    assert ocr_extract.pdf_to_text(Path("any.pdf")) == "text of img1\n\ntext of img2"


def test_extract_all_creates_output_dir(tmp_path):
    """extract_all should create the output directory if it does not exist."""
    from src.pipeline.ocr_extract import extract_all

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    # With no PDFs it should not fail
    count = extract_all(input_dir=input_dir, output_dir=output_dir)
    assert count == 0
    assert output_dir.exists()


def test_extract_all_skips_already_processed(tmp_path, monkeypatch):
    """A PDF whose .txt already exists is not reprocessed (resumable pipeline)."""
    from src.pipeline import ocr_extract

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "paper1.pdf").write_bytes(b"%PDF-1.4")
    (input_dir / "paper2.pdf").write_bytes(b"%PDF-1.4")
    (output_dir / "paper1.txt").write_text("already processed", encoding="utf-8")

    monkeypatch.setattr(ocr_extract, "pdf_to_text", lambda p, lang="eng": "new text")

    count = ocr_extract.extract_all(input_dir=input_dir, output_dir=output_dir)

    assert count == 1
    assert (output_dir / "paper1.txt").read_text(encoding="utf-8") == "already processed"
    assert (output_dir / "paper2.txt").read_text(encoding="utf-8") == "new text"
