"""
ocr_extract.py — Extracts text from PDFs with Tesseract OCR, converting each
PDF to images first.
"""

import os
import logging
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
OCR_DIR = Path(os.getenv("OCR_DIR", "data/ocr"))


def pdf_to_text(pdf_path: Path, lang: str = "eng") -> str:
    """Extract text from a PDF using Tesseract, one OCR pass per page."""
    images = convert_from_path(pdf_path, dpi=300)
    pages = [pytesseract.image_to_string(img, lang=lang) for img in images]
    return "\n\n".join(pages)


def extract_all(input_dir: Path | None = None, output_dir: Path | None = None) -> int:
    """
    OCR every PDF in input_dir into output_dir, skipping ones already done.
    Returns the number of files processed.
    """
    input_dir = input_dir or RAW_DIR
    output_dir = output_dir or OCR_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    pdf_files = sorted(input_dir.glob("*.pdf"))
    total = len(pdf_files)

    for i, pdf_file in enumerate(pdf_files, 1):
        output_path = output_dir / f"{pdf_file.stem}.txt"
        if output_path.exists():
            print(f"[{i}/{total}] Skipping (already processed): {pdf_file.name}")
            continue

        print(f"[{i}/{total}] Processing: {pdf_file.name} ...", end=" ", flush=True)
        try:
            text = pdf_to_text(pdf_file)
            output_path.write_text(text, encoding="utf-8")
            count += 1
            print("OK")
            logger.info(f"OCR completed: {pdf_file.name}")
        except Exception as e:
            print(f"ERROR: {e}")
            logger.error(f"Error processing {pdf_file.name}: {e}")

    return count


if __name__ == "__main__":
    from src.utils.pipeline_stats import StepTimer, print_summary
    import os

    with StepTimer("3_ocr_extract") as t:
        processed = extract_all()
        n_pdfs = len(list(RAW_DIR.glob("*.pdf")))
        t.record("n_pdfs_total", n_pdfs)
        t.record("n_processed", processed)
        t.record("dpi", 300)
        total_pages = sum(
            int(__import__('subprocess').check_output(
                ['pdfinfo', str(p)], stderr=__import__('subprocess').DEVNULL
            ).decode().split('Pages:')[1].split()[0])
            for p in RAW_DIR.glob('*.pdf')
        )
        t.record("total_pages", total_pages)
        t.record("avg_sec_per_page", round(t.elapsed / max(total_pages, 1), 2))

    print(f"Processed {processed} PDF files.")
    print_summary()
