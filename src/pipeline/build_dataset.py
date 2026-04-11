"""
build_dataset.py — Orquesta el pipeline completo:
1. Fetch de OpenAlex
2. Descarga de PDFs
3. Extracción OCR
4. Alineación de texto
5. Generación de pares de entrenamiento
"""

import json
import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.pipeline.fetch_openalex import fetch_works, save_abstracts
from src.pipeline.download_pdfs import download_all_pdfs
from src.pipeline.ocr_extract import extract_all
from src.pipeline.align_text import align_files
from src.utils.logger import setup_logger

load_dotenv()
logger = logging.getLogger(__name__)

PAIRS_DIR = Path(os.getenv("PAIRS_DIR", "data/pairs"))
OCR_DIR = Path(os.getenv("OCR_DIR", "data/ocr"))
GROUND_TRUTH_DIR = Path(os.getenv("GROUND_TRUTH_DIR", "data/ground_truth"))


def build_pairs(ocr_dir: Path, gt_dir: Path, output_dir: Path) -> int:
    """
    Construye pares alineados OCR ↔ ground truth para todos los archivos.

    Returns:
        Número total de pares generados.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_pairs = []

    for ocr_file in ocr_dir.glob("*.txt"):
        gt_file = gt_dir / ocr_file.name
        if not gt_file.exists():
            logger.warning(f"Sin ground truth para: {ocr_file.name}")
            continue

        pairs = align_files(ocr_file, gt_file)
        all_pairs.extend(pairs)
        logger.info(f"{ocr_file.name}: {len(pairs)} pares encontrados")

    output_path = output_dir / "dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    logger.info(f"Dataset guardado: {output_path} ({len(all_pairs)} pares)")
    return len(all_pairs)


def run_pipeline(config_path: str = "configs/openalex_query.yaml") -> None:
    """Ejecuta el pipeline completo."""
    setup_logger()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("=== Paso 1: Fetch de OpenAlex ===")
    works = fetch_works(config)
    save_abstracts(works)

    logger.info("=== Paso 2: Descarga de PDFs ===")
    stats = download_all_pdfs(works)
    logger.info(f"Descarga: {stats}")

    logger.info("=== Paso 3: Extracción OCR ===")
    count = extract_all()
    logger.info(f"OCR completado: {count} archivos")

    logger.info("=== Paso 4: Alineación y generación de pares ===")
    total_pairs = build_pairs(OCR_DIR, GROUND_TRUTH_DIR, PAIRS_DIR)
    logger.info(f"Pipeline completo. Total de pares: {total_pairs}")


if __name__ == "__main__":
    run_pipeline()
