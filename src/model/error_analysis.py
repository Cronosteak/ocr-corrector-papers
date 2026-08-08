"""
error_analysis.py — Categorizes the model's residual errors.

Compares each prediction against the ground truth token by token and sorts every
missed token into a category (numbers, acronyms, non-ASCII symbols, punctuation,
content words), showing *where* the model still fails.

Run standalone:
    python -m src.model.error_analysis --predictions models/<run>/predictions.json
or import it from postprocess.py.
"""

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")
_GREEK_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _categorize_token(tok: str) -> str:
    """Assign a category to a ground-truth token the model got wrong."""
    if _GREEK_RE.search(tok):
        return "greek_symbol"
    if _NON_ASCII_RE.search(tok):
        return "non_ascii"
    if any(c.isdigit() for c in tok):
        return "number"
    if not tok.isalnum() and not tok.isspace():
        return "punctuation"
    if tok.isupper() and len(tok) > 1:
        return "acronym"
    if tok[:1].isupper():
        return "capitalized_word"
    if tok.isalpha():
        return "lowercase_word"
    return "other"


def _diff_categories(pred: str, gt: str) -> Counter:
    """
    Fast multiset diff: count ground-truth tokens that do not appear (at the same
    frequency) in the prediction. Not a true alignment, but it captures the error
    pattern well.
    """
    pred_tokens = Counter(_tokenize(pred))
    gt_tokens = Counter(_tokenize(gt))
    missing = gt_tokens - pred_tokens  # tokens in the GT that were not recovered
    cats = Counter()
    for tok, n in missing.items():
        cats[_categorize_token(tok)] += n
    return cats


def analyze(predictions: list[dict]) -> dict:
    """
    Aggregate error counts per category and per method.

    predictions: dicts with keys ocr, spellcheck, corrected, ground_truth, noise_rate.
    """
    methods = ["ocr", "spellcheck", "corrected"]
    total_errors = {m: Counter() for m in methods}
    samples_with_error = {m: Counter() for m in methods}
    per_noise = defaultdict(lambda: {m: Counter() for m in methods})

    for ex in predictions:
        gt = ex["ground_truth"]
        rate = ex.get("noise_rate")
        for m in methods:
            cats = _diff_categories(ex[m], gt)
            for c, n in cats.items():
                total_errors[m][c] += n
            for c in cats:
                samples_with_error[m][c] += 1
            if rate is not None:
                for c, n in cats.items():
                    per_noise[rate][m][c] += n

    all_cats = sorted({c for m in methods for c in total_errors[m]})
    summary = {
        "categories": all_cats,
        "total_errors_by_method": {m: dict(total_errors[m]) for m in methods},
        "samples_affected_by_method": {m: dict(samples_with_error[m]) for m in methods},
        "per_noise_rate": {
            f"r={r:.2f}": {m: dict(per_noise[r][m]) for m in methods}
            for r in sorted(per_noise)
        },
    }
    return summary


def plot_error_categories(summary: dict, out_dir: Path) -> None:
    cats = summary["categories"]
    methods = ["ocr", "spellcheck", "corrected"]
    labels = {"ocr": "Raw OCR", "spellcheck": "Spellchecker", "corrected": "Ours (flan-t5)"}
    colors = {"ocr": "#e74c3c", "spellcheck": "#f39c12", "corrected": "#2ecc71"}

    x = list(range(len(cats)))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(methods):
        vals = [summary["total_errors_by_method"][m].get(c, 0) for c in cats]
        ax.bar([xi + (i - 1) * w for xi in x], vals, width=w,
               label=labels[m], color=colors[m])
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylabel("Number of missed tokens (lower is better)")
    ax.set_title("Residual error categories by method (full test set)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = out_dir / "error_categories.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Saved: {out}")


def run_error_analysis(predictions: list[dict], out_dir: Path) -> dict:
    summary = analyze(predictions)
    out_json = out_dir / "error_analysis.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {out_json}")
    plot_error_categories(summary, out_dir)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                        help="Path to predictions.json produced by postprocess.py")
    args = parser.parse_args()
    p = Path(args.predictions)
    preds = json.load(open(p))
    run_error_analysis(preds, p.parent)
