"""
metrics.py — Funciones para calcular métricas de calidad OCR.
CER (Character Error Rate), WER (Word Error Rate), BLEU.
"""

from jiwer import wer, cer


def calculate_cer(predictions: list[str], references: list[str]) -> float:
    """
    Calcula el Character Error Rate promedio.

    Args:
        predictions: Lista de textos predichos/OCR.
        references: Lista de textos de referencia (ground truth).

    Returns:
        CER promedio (0.0 = perfecto, 1.0 = todo incorrecto).
    """
    return cer(references, predictions)


def calculate_wer(predictions: list[str], references: list[str]) -> float:
    """
    Calcula el Word Error Rate promedio.

    Args:
        predictions: Lista de textos predichos/OCR.
        references: Lista de textos de referencia (ground truth).

    Returns:
        WER promedio.
    """
    return wer(references, predictions)


def calculate_improvement(
    baseline_metric: float, corrected_metric: float
) -> dict:
    """
    Calcula la mejora relativa entre baseline y corregido.

    Args:
        baseline_metric: Métrica antes de la corrección.
        corrected_metric: Métrica después de la corrección.

    Returns:
        Diccionario con absolute_improvement y relative_improvement_pct.
    """
    absolute = baseline_metric - corrected_metric
    relative_pct = (absolute / baseline_metric * 100) if baseline_metric > 0 else 0.0

    return {
        "baseline": round(baseline_metric, 4),
        "corrected": round(corrected_metric, 4),
        "absolute_improvement": round(absolute, 4),
        "relative_improvement_pct": round(relative_pct, 2),
    }
