"""
ocr_extract.py — Extrae texto de PDFs usando Tesseract OCR.
Convierte cada PDF a imágenes y luego aplica OCR.
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
    """
    Extrae texto de un PDF usando Tesseract.

    Args:
        pdf_path: Ruta al archivo PDF.
        lang: Idioma para Tesseract (por defecto inglés).

    Returns:
        Texto extraído del PDF.
    """
    images = convert_from_path(pdf_path, dpi=300)
    pages = [pytesseract.image_to_string(img, lang=lang) for img in images]
    return "\n\n".join(pages)


def extract_all(input_dir: Path | None = None, output_dir: Path | None = None) -> int:
    """
    Procesa todos los PDFs en input_dir y guarda el texto OCR en output_dir.

    Args:
        input_dir: Directorio con PDFs.
        output_dir: Directorio donde guardar texto OCR.

    Returns:
        Número de archivos procesados.
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
            print(f"[{i}/{total}] Saltando (ya procesado): {pdf_file.name}")
            continue

        print(f"[{i}/{total}] Procesando: {pdf_file.name} ...", end=" ", flush=True)
        try:
            text = pdf_to_text(pdf_file)
            output_path.write_text(text, encoding="utf-8")
            count += 1
            print("OK")
            logger.info(f"OCR completado: {pdf_file.name}")
        except Exception as e:
            print(f"ERROR: {e}")
            logger.error(f"Error procesando {pdf_file.name}: {e}")

    return count


if __name__ == "__main__":
    from src.utils.pipeline_stats import StepTimer, print_summary
    import os

    with StepTimer("3_ocr_extract") as t:
        processed = extract_all()
        n_pdfs = len(list(RAW_DIR.glob("*.pdf")))
        t.record("n_pdfs_total", n_pdfs)
        t.record("n_processed", processed)
        t.record("dpi", 200)
        total_pages = sum(
            int(__import__('subprocess').check_output(
                ['pdfinfo', str(p)], stderr=__import__('subprocess').DEVNULL
            ).decode().split('Pages:')[1].split()[0])
            for p in RAW_DIR.glob('*.pdf')
        )
        t.record("total_pages", total_pages)
        t.record("avg_sec_per_page", round(t.__dict__.get('_elapsed', 0) / max(total_pages, 1), 2))

    print(f"Se procesaron {processed} archivos PDF.")
    print_summary()
