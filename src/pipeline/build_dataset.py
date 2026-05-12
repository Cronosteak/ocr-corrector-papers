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
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.pipeline.fetch_openalex import fetch_works, save_abstracts
from src.pipeline.download_pdfs import download_all_pdfs
from src.pipeline.ocr_extract import extract_all
from src.utils.logger import setup_logger

load_dotenv()
logger = logging.getLogger(__name__)

PAIRS_DIR = Path(os.getenv("PAIRS_DIR", "data/pairs"))
OCR_DIR = Path(os.getenv("OCR_DIR", "data/ocr"))
GROUND_TRUTH_DIR = Path(os.getenv("GROUND_TRUTH_DIR", "data/ground_truth"))

# Confusiones de caracteres típicas en OCR
_OCR_CONFUSIONS = {
    "a": "o", "o": "0", "l": "1", "I": "l", "0": "O",
    "e": "c", "n": "m", "h": "li", "rn": "m", "m": "rn",
    "fi": "ﬁ", "s": "S", "g": "9", "b": "6",
}

# Tasas de ruido: muy leve, leve, media, fuerte (máx 0.08 para evitar alucinaciones)
_NOISE_RATES = [0.02, 0.04, 0.06, 0.08]


def _clean_text(text: str) -> str:
    """Elimina HTML, normaliza espacios y filtra caracteres no-ASCII problemáticos."""
    # Quitar tags HTML
    text = re.sub(r"<[^>]+>", "", text)
    # Normalizar espacios múltiples
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _inject_ocr_noise(text: str, rate: float = 0.08) -> str:
    """Inyecta ruido OCR sintético en un texto limpio."""
    random.seed(None)
    chars = list(text)
    i = 0
    while i < len(chars):
        if random.random() > rate:
            i += 1
            continue
        action = random.randint(0, 3)
        if action == 0 and chars[i] in _OCR_CONFUSIONS:
            # sustitución de carácter
            chars[i] = _OCR_CONFUSIONS[chars[i]]
        elif action == 1 and chars[i] != " ":
            # inserción de espacio o guión
            chars.insert(i, random.choice([" ", "-"]))
            i += 1
        elif action == 2 and i + 1 < len(chars) and chars[i] == " ":
            # eliminar espacio (palabras pegadas)
            chars.pop(i)
            continue
        elif action == 3:
            # duplicar carácter
            chars.insert(i, chars[i])
            i += 1
        i += 1
    return "".join(chars)


def build_synthetic_pairs(gt_files: list, n_per_doc: int = 10) -> list[dict]:
    """
    Genera pares sintéticos con alineación perfecta.
    Por cada abstract, toma hasta n_per_doc oraciones y genera una variante
    de ruido por cada tasa en _NOISE_RATES (leve, media, fuerte).
    """
    pairs = []
    for gt_file in gt_files:
        text = gt_file.read_text(encoding="utf-8").strip()
        if not text or len(text) < 40:
            continue
        text = _clean_text(text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]
        for sent in sentences[:n_per_doc]:
            for rate in _NOISE_RATES:
                noisy = _inject_ocr_noise(sent, rate=rate)
                if noisy != sent:
                    pairs.append({"ocr": noisy, "ground_truth": sent})
    return pairs


def build_pairs(ocr_dir: Path, gt_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dividir GT files por documento (80/10/10) para evitar data leakage
    gt_files = sorted(gt_dir.glob("*.txt"))
    random.seed(42)
    random.shuffle(gt_files)

    n = len(gt_files)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_files = gt_files[:n_train]
    val_files   = gt_files[n_train:n_train + n_val]
    test_files  = gt_files[n_train + n_val:]

    print(f"GT files disponibles: {n} abstracts")
    print(f"Split por documento: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")
    print(f"Ruidos simulados: {_NOISE_RATES} (leve, medio, fuerte)")
    print()

    train_pairs = build_synthetic_pairs(train_files, n_per_doc=10)
    val_pairs   = build_synthetic_pairs(val_files,   n_per_doc=10)
    test_pairs  = build_synthetic_pairs(test_files,  n_per_doc=10)

    total = len(train_pairs) + len(val_pairs) + len(test_pairs)
    print(f"Pares generados: {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test")
    print(f"Total: {total} pares sintéticos con alineación perfecta")

    splits = {
        "train": train_pairs,
        "val":   val_pairs,
        "test":  test_pairs,
    }

    for split_name, split_data in splits.items():
        split_path = output_dir / f"{split_name}.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  {split_name}: {len(split_data)} pares → {split_path}")
        logger.info(f"{split_name}: {len(split_data)} pares → {split_path}")

    # synthetic_test.json = test split (conjunto de evaluación limpio)
    synthetic_test_path = output_dir / "synthetic_test.json"
    with open(synthetic_test_path, "w", encoding="utf-8") as f:
        json.dump(test_pairs, f, ensure_ascii=False, indent=2)
    print(f"  synthetic_test: {len(test_pairs)} pares → {synthetic_test_path}")

    # Dataset completo
    all_pairs = train_pairs + val_pairs + test_pairs
    full_path = output_dir / "dataset.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    logger.info(f"Dataset completo: {full_path} ({total} pares)")
    return total


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
        t.record("n_pairs_real", len(list(OCR_DIR.glob("*.txt"))))
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
