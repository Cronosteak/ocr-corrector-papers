"""
build_dataset.py — Orchestrates the full pipeline:
1. Fetch from OpenAlex
2. Download PDFs
3. OCR extraction
4. Text alignment
5. Training pair generation
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

# Character confusions typical of OCR
_OCR_CONFUSIONS = {
    "a": "o", "o": "0", "l": "1", "I": "l", "0": "O",
    "e": "c", "n": "m", "h": "li", "rn": "m", "m": "rn",
    "fi": "ﬁ", "s": "S", "g": "9", "b": "6",
}

# Noise rates: very light to heavy (capped at 0.08 to avoid hallucinations)
_NOISE_RATES = [0.02, 0.04, 0.06, 0.08]


def _clean_text(text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _inject_ocr_noise(text: str, rate: float = 0.08, seed: int | None = None) -> str:
    """Inject synthetic OCR noise into a clean text."""
    rng = random.Random(seed)
    chars = list(text)
    i = 0
    while i < len(chars):
        if rng.random() > rate:
            i += 1
            continue
        action = rng.randint(0, 3)
        if action == 0 and chars[i] in _OCR_CONFUSIONS:
            # character substitution
            chars[i] = _OCR_CONFUSIONS[chars[i]]
        elif action == 1 and chars[i] != " ":
            # insert a space or hyphen
            chars.insert(i, rng.choice([" ", "-"]))
            i += 1
        elif action == 2 and i + 1 < len(chars) and chars[i] == " ":
            # delete a space (words run together)
            chars.pop(i)
            continue
        elif action == 3:
            # duplicate the character
            chars.insert(i, chars[i])
            i += 1
        i += 1
    return "".join(chars)


def build_synthetic_pairs(gt_files: list, n_per_doc: int = 10) -> list[dict]:
    """
    Generate synthetic pairs with perfect alignment. For each abstract, take up
    to n_per_doc sentences and produce one noisy variant per rate in _NOISE_RATES.
    """
    pairs = []
    pair_counter = 0
    for gt_file in gt_files:
        text = gt_file.read_text(encoding="utf-8").strip()
        if not text or len(text) < 40:
            continue
        text = _clean_text(text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]
        for sent in sentences[:n_per_doc]:
            for rate in _NOISE_RATES:
                noisy = _inject_ocr_noise(sent, rate=rate, seed=pair_counter)
                pair_counter += 1
                if noisy != sent:
                    pairs.append({
                        "ocr": noisy,
                        "ground_truth": sent,
                        "noise_rate": rate,
                    })
    return pairs


def build_pairs(ocr_dir: Path, gt_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split GT files per document (80/10/10) to avoid data leakage
    gt_files = sorted(gt_dir.glob("*.txt"))
    random.seed(42)
    random.shuffle(gt_files)

    n = len(gt_files)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_files = gt_files[:n_train]
    val_files   = gt_files[n_train:n_train + n_val]
    test_files  = gt_files[n_train + n_val:]

    print(f"GT files available: {n} abstracts")
    print(f"Per-document split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")
    print(f"Simulated noise rates: {_NOISE_RATES}")
    print()

    train_pairs = build_synthetic_pairs(train_files, n_per_doc=10)
    val_pairs   = build_synthetic_pairs(val_files,   n_per_doc=10)
    test_pairs  = build_synthetic_pairs(test_files,  n_per_doc=10)

    total = len(train_pairs) + len(val_pairs) + len(test_pairs)
    print(f"Pairs generated: {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test")
    print(f"Total: {total} synthetic pairs with perfect alignment")

    splits = {
        "train": train_pairs,
        "val":   val_pairs,
        "test":  test_pairs,
    }

    for split_name, split_data in splits.items():
        split_path = output_dir / f"{split_name}.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  {split_name}: {len(split_data)} pairs → {split_path}")
        logger.info(f"{split_name}: {len(split_data)} pairs → {split_path}")

    # synthetic_test.json = test split (clean evaluation set)
    synthetic_test_path = output_dir / "synthetic_test.json"
    with open(synthetic_test_path, "w", encoding="utf-8") as f:
        json.dump(test_pairs, f, ensure_ascii=False, indent=2)
    print(f"  synthetic_test: {len(test_pairs)} pairs → {synthetic_test_path}")

    # Full dataset
    all_pairs = train_pairs + val_pairs + test_pairs
    full_path = output_dir / "dataset.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    logger.info(f"Full dataset: {full_path} ({total} pairs)")
    return total


def run_pipeline(config_path: str = "configs/openalex_query.yaml") -> None:
    """Run the full pipeline end to end."""
    setup_logger()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("=== Step 1: Fetch from OpenAlex ===")
    works = fetch_works(config)
    save_abstracts(works)

    logger.info("=== Step 2: Download PDFs ===")
    stats = download_all_pdfs(works)
    logger.info(f"Download: {stats}")

    logger.info("=== Step 3: OCR extraction ===")
    count = extract_all()
    logger.info(f"OCR completed: {count} files")

    logger.info("=== Step 4: Alignment and pair generation ===")
    total_pairs = build_pairs(OCR_DIR, GROUND_TRUTH_DIR, PAIRS_DIR)
    logger.info(f"Pipeline complete. Total pairs: {total_pairs}")


if __name__ == "__main__":
    from src.utils.pipeline_stats import StepTimer, print_summary

    with StepTimer("4_build_dataset") as t:
        total = build_pairs(OCR_DIR, GROUND_TRUTH_DIR, PAIRS_DIR)
        t.record("n_pairs_total", total)
        t.record("n_pairs_real", len(list(OCR_DIR.glob("*.txt"))))
        t.record("n_docs", len(list(OCR_DIR.glob("*.txt"))))
        t.record("min_similarity_threshold", 0.4)

        # Read back the generated splits
        import json
        for split in ["train", "val", "test"]:
            p = PAIRS_DIR / f"{split}.json"
            if p.exists():
                n = len(json.load(open(p)))
                t.record(f"n_{split}", n)

    print(f"\nDataset complete: {total} pairs total.")
    print_summary()
