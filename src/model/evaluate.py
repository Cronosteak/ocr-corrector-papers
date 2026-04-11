"""
evaluate.py — Evalúa el modelo de corrección OCR calculando
CER (Character Error Rate) y WER (Word Error Rate) antes y después de la corrección.
"""

import json
import logging
from pathlib import Path

from src.utils.metrics import calculate_cer, calculate_wer

logger = logging.getLogger(__name__)


def evaluate_model(
    pairs_path: str = "data/pairs/dataset.json",
    model_path: str | None = None,
) -> dict:
    """
    Evalúa el modelo comparando métricas antes y después de la corrección.

    Args:
        pairs_path: Ruta al dataset de pares.
        model_path: Ruta al modelo entrenado. Si None, solo calcula baseline.

    Returns:
        Diccionario con métricas {baseline_cer, baseline_wer, corrected_cer, corrected_wer}.
    """
    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    # Baseline: métricas del OCR original vs ground truth
    ocr_texts = [p["ocr"] for p in pairs]
    gt_texts = [p["ground_truth"] for p in pairs]

    baseline_cer = calculate_cer(ocr_texts, gt_texts)
    baseline_wer = calculate_wer(ocr_texts, gt_texts)

    results = {
        "baseline_cer": baseline_cer,
        "baseline_wer": baseline_wer,
        "num_samples": len(pairs),
    }

    if model_path:
        # TODO: Cargar modelo y generar correcciones
        # corrected_texts = predict_batch(model_path, ocr_texts)
        # results["corrected_cer"] = calculate_cer(corrected_texts, gt_texts)
        # results["corrected_wer"] = calculate_wer(corrected_texts, gt_texts)
        logger.warning("Evaluación post-modelo aún no implementada")

    logger.info(f"Resultados: {results}")
    return results


if __name__ == "__main__":
    results = evaluate_model()
    print(json.dumps(results, indent=2))
