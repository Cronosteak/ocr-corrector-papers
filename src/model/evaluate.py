"""
evaluate.py — Evalúa el modelo de corrección OCR calculando
CER (Character Error Rate) y WER (Word Error Rate) antes y después de la corrección.
"""

import json
import logging
from pathlib import Path

from src.utils.metrics import calculate_cer, calculate_improvement, calculate_wer

logger = logging.getLogger(__name__)


def evaluate_model(
    pairs_path: str = "data/pairs/synthetic_test.json",
    model_path: str | None = None,
) -> dict:
    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    ocr_texts = [p["ocr"] for p in pairs]
    gt_texts = [p["ground_truth"] for p in pairs]

    baseline_cer = calculate_cer(ocr_texts, gt_texts)
    baseline_wer = calculate_wer(ocr_texts, gt_texts)

    results = {
        "num_samples": len(pairs),
        "baseline_cer": round(baseline_cer, 4),
        "baseline_wer": round(baseline_wer, 4),
    }

    if model_path:
        from src.model.predict import correct_batch, load_model

        logger.info(f"Cargando modelo desde {model_path}...")
        model, tokenizer = load_model(model_path)

        logger.info(f"Generando correcciones para {len(ocr_texts)} ejemplos...")
        corrected_texts = correct_batch(ocr_texts, model, tokenizer, batch_size=16)

        corrected_cer = calculate_cer(corrected_texts, gt_texts)
        corrected_wer = calculate_wer(corrected_texts, gt_texts)

        results["cer"] = calculate_improvement(baseline_cer, corrected_cer)
        results["wer"] = calculate_improvement(baseline_wer, corrected_wer)

    logger.info(f"Resultados: {results}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Ruta al modelo entrenado")
    parser.add_argument("--data", type=str, default="data/pairs/synthetic_test.json")
    args = parser.parse_args()

    results = evaluate_model(pairs_path=args.data, model_path=args.model)
    print(json.dumps(results, indent=2))
