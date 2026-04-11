"""
ocr_extract.py — Extrae texto de PDFs usando Tesseract OCR.
Convierte cada PDF a imágenes y luego aplica OCR.
"""

import os
import logging
from pathlib import Path

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
    # TODO: Implementar con pdf2image + pytesseract
    # 1. Convertir PDF a lista de imágenes (pdf2image.convert_from_path)
    # 2. Aplicar pytesseract.image_to_string a cada imagen
    # 3. Concatenar texto
    raise NotImplementedError("Implementar extracción OCR con Tesseract")


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
    for pdf_file in input_dir.glob("*.pdf"):
        output_path = output_dir / f"{pdf_file.stem}.txt"
        if output_path.exists():
            continue

        try:
            text = pdf_to_text(pdf_file)
            output_path.write_text(text, encoding="utf-8")
            count += 1
            logger.info(f"OCR completado: {pdf_file.name}")
        except Exception as e:
            logger.error(f"Error procesando {pdf_file.name}: {e}")

    return count


if __name__ == "__main__":
    processed = extract_all()
    print(f"Se procesaron {processed} archivos PDF.")
