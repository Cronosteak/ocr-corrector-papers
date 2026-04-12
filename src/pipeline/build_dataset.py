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
import random
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

    ocr_files = sorted(ocr_dir.glob("*.txt"))
    total = len(ocr_files)
    print(f"Alineando {total} documentos...")

    for i, ocr_file in enumerate(ocr_files, 1):
        gt_file = gt_dir / ocr_file.name
        if not gt_file.exists():
            print(f"[{i}/{total}] Sin ground truth: {ocr_file.name} — saltando")
            continue

        pairs = align_files(ocr_file, gt_file)
        all_pairs.extend(pairs)
        print(f"[{i}/{total}] {ocr_file.stem}: {len(pairs)} pares  (total acumulado: {len(all_pairs)})")

    # Split 80/10/10
    random.seed(42)
    random.shuffle(all_pairs)
    n = len(all_pairs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    splits = {
        "train": all_pairs[:n_train],
        "val": all_pairs[n_train:n_train + n_val],
        "test": all_pairs[n_train + n_val:],
    }

    print(f"\n--- Split 80/10/10 ---")
    for split_name, split_data in splits.items():
        split_path = output_dir / f"{split_name}.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  {split_name}: {len(split_data)} pares → {split_path}")
        logger.info(f"{split_name}: {len(split_data)} pares → {split_path}")

    # También guardar dataset completo
    full_path = output_dir / "dataset.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    logger.info(f"Dataset completo: {full_path} ({n} pares)")
    return n


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
    from src.utils.pipeline_stats import StepTimer, print_summary

    with StepTimer("4_build_dataset") as t:
        total = build_pairs(OCR_DIR, GROUND_TRUTH_DIR, PAIRS_DIR)
        t.record("n_pairs_total", total)
        t.record("n_docs", len(list(OCR_DIR.glob("*.txt"))))
        t.record("min_similarity_threshold", 0.4)

        # Leer splits generados
        import json
        for split in ["train", "val", "test"]:
            p = PAIRS_DIR / f"{split}.json"
            if p.exists():
                n = len(json.load(open(p)))
                t.record(f"n_{split}", n)

    print(f"\nDataset completo: {total} pares totales.")
    print_summary()
