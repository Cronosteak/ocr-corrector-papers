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
from src.pipeline.align_text import align_files
from src.utils.logger import setup_logger

load_dotenv()
logger = logging.getLogger(__name__)

PAIRS_DIR = Path(os.getenv("PAIRS_DIR", "data/pairs"))
OCR_DIR = Path(os.getenv("OCR_DIR", "data/ocr"))
GROUND_TRUTH_DIR = Path(os.getenv("GROUND_TRUTH_DIR", "data/ground_truth"))

# Confuciones de caracteres típicas en OCR
_OCR_CONFUSIONS = {
    "a": "o", "o": "0", "l": "1", "I": "l", "0": "O",
    "e": "c", "n": "m", "h": "li", "rn": "m", "m": "rn",
    "fi": "ﬁ", "s": "S", "g": "9", "b": "6",
}

_PUNCT_NOISE = [" ", "-", "", ","]


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


def build_synthetic_pairs(gt_dir: Path, n_per_doc: int = 5) -> list[dict]:
    """
    Genera pares sintéticos: abstract limpio → versión con ruido OCR simulado.
    Produce pares de alta calidad con alineación perfecta.
    """
    pairs = []
    gt_files = sorted(gt_dir.glob("*.txt"))
    for gt_file in gt_files:
        text = gt_file.read_text(encoding="utf-8").strip()
        if not text or len(text) < 40:
            continue
        # Dividir en oraciones
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]
        for sent in sentences[:n_per_doc]:
            noisy = _inject_ocr_noise(sent, rate=0.08)
            if noisy != sent:
                pairs.append({"ocr": noisy, "ground_truth": sent})
    return pairs


def build_pairs(ocr_dir: Path, gt_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_pairs = []

    # Pares reales (OCR extraído ↔ abstract)
    ocr_files = sorted(ocr_dir.glob("*.txt"))
    total = len(ocr_files)
    print(f"Alineando {total} documentos (pares reales)...")

    for i, ocr_file in enumerate(ocr_files, 1):
        gt_file = gt_dir / ocr_file.name
        if not gt_file.exists():
            print(f"[{i}/{total}] Sin ground truth: {ocr_file.name} — saltando")
            continue
        pairs = align_files(ocr_file, gt_file)
        all_pairs.extend(pairs)
        print(f"[{i}/{total}] {ocr_file.stem}: {len(pairs)} pares  (total acumulado: {len(all_pairs)})")

    real_count = len(all_pairs)

    # Pares sintéticos (abstract + ruido OCR simulado)
    print(f"\nGenerando pares sintéticos desde {len(list(gt_dir.glob('*.txt')))} abstracts...")
    synthetic = build_synthetic_pairs(gt_dir, n_per_doc=5)
    print(f"Pares sintéticos generados: {len(synthetic)}")

    # Split sintético separado 80/10/10 → para evaluación limpia (paper)
    random.seed(42)
    random.shuffle(synthetic)
    n_s = len(synthetic)
    n_s_train = int(n_s * 0.8)
    n_s_val = int(n_s * 0.1)
    synthetic_splits = {
        "train": synthetic[:n_s_train],
        "val": synthetic[n_s_train:n_s_train + n_s_val],
        "test": synthetic[n_s_train + n_s_val:],
    }
    for split_name, split_data in synthetic_splits.items():
        path = output_dir / f"synthetic_{split_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  synthetic_{split_name}: {len(split_data)} pares → {path}")

    # Combinar para entrenamiento
    all_pairs.extend(synthetic)
    print(f"Total combinado: {len(all_pairs)} pares ({real_count} reales + {len(synthetic)} sintéticos)")

    # Split 80/10/10 combinado
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

    print(f"\n--- Split 80/10/10 combinado ---")
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
