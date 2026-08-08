"""
metrics.py — OCR quality metrics: CER, WER, BLEU, exact match.
"""

from jiwer import wer, cer
import sacrebleu


def calculate_cer(predictions: list[str], references: list[str]) -> float:
    """Average Character Error Rate (0.0 = perfect, 1.0 = everything wrong)."""
    return cer(references, predictions)


def calculate_wer(predictions: list[str], references: list[str]) -> float:
    """Average Word Error Rate."""
    return wer(references, predictions)


def calculate_bleu(predictions: list[str], references: list[str]) -> float:
    """Corpus-level BLEU via sacrebleu (0-100, higher is better)."""
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score


def calculate_exact_match(predictions: list[str], references: list[str]) -> float:
    """Percentage of predictions identical to the ground truth."""
    if not predictions:
        return 0.0
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return matches / len(predictions) * 100


def calculate_improvement(
    baseline_metric: float, corrected_metric: float
) -> dict:
    """Absolute and relative improvement of a corrected metric over its baseline."""
    absolute = baseline_metric - corrected_metric
    relative_pct = (absolute / baseline_metric * 100) if baseline_metric > 0 else 0.0

    return {
        "baseline": round(baseline_metric, 4),
        "corrected": round(corrected_metric, 4),
        "absolute_improvement": round(absolute, 4),
        "relative_improvement_pct": round(relative_pct, 2),
    }
