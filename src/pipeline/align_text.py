"""
align_text.py — Alinea texto OCR (ruidoso) con el texto ground truth (limpio)
usando distancia de edición para crear pares de entrenamiento.
"""

import logging
from pathlib import Path

from Levenshtein import ratio as levenshtein_ratio

logger = logging.getLogger(__name__)


def align_paragraphs(
    ocr_text: str, gt_text: str, min_similarity: float = 0.3
) -> list[dict]:
    """
    Alinea párrafos del texto OCR con párrafos del ground truth.

    Args:
        ocr_text: Texto completo extraído por OCR.
        gt_text: Texto limpio (ground truth).
        min_similarity: Umbral mínimo de similitud para considerar un par válido.

    Returns:
        Lista de pares {ocr_paragraph, gt_paragraph, similarity}.
    """
    ocr_paragraphs = [p.strip() for p in ocr_text.split("\n\n") if p.strip()]
    gt_paragraphs = [p.strip() for p in gt_text.split("\n\n") if p.strip()]

    pairs = []
    for ocr_para in ocr_paragraphs:
        best_match = None
        best_score = 0.0

        for gt_para in gt_paragraphs:
            score = levenshtein_ratio(ocr_para, gt_para)
            if score > best_score:
                best_score = score
                best_match = gt_para

        if best_match and best_score >= min_similarity:
            pairs.append(
                {
                    "ocr": ocr_para,
                    "ground_truth": best_match,
                    "similarity": round(best_score, 4),
                }
            )

    return pairs


def align_files(ocr_path: Path, gt_path: Path) -> list[dict]:
    """
    Alinea un archivo OCR con su correspondiente ground truth.

    Args:
        ocr_path: Ruta al archivo de texto OCR.
        gt_path: Ruta al archivo de texto ground truth.

    Returns:
        Lista de pares alineados.
    """
    ocr_text = ocr_path.read_text(encoding="utf-8")
    gt_text = gt_path.read_text(encoding="utf-8")
    return align_paragraphs(ocr_text, gt_text)
