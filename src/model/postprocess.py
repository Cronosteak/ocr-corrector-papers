import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required on cluster
import matplotlib.pyplot as plt
from jiwer import cer as compute_cer

from src.model.baselines import spellcheck_correct
from src.model.predict import correct_batch, load_model
from src.utils.metrics import (
    calculate_bleu,
    calculate_cer,
    calculate_exact_match,
    calculate_improvement,
    calculate_wer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def plot_training_curves(history_path: Path, out_dir: Path) -> None:
    history = json.load(open(history_path))

    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    for entry in history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_loss.append(entry["eval_loss"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_steps, train_loss, label="Train loss", linewidth=1.5)
    ax.plot(eval_steps, eval_loss, label="Validation loss", linewidth=1.5, marker="o", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training curves — flan-t5-base OCR corrector")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "training_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Saved: {out}")


def _metric_block(preds, refs):
    return {
        "cer": round(calculate_cer(preds, refs), 4),
        "wer": round(calculate_wer(preds, refs), 4),
        "bleu": round(calculate_bleu(preds, refs), 2),
        "exact_match_pct": round(calculate_exact_match(preds, refs), 2),
    }


def evaluate_all(test_pairs, model_path: str):
    """
    Evaluate the raw-OCR baseline, the spellchecker baseline and the model.
    Returns overall metrics, per-noise-level metrics and the generated texts.
    """
    ocr_texts = [p["ocr"] for p in test_pairs]
    gt_texts  = [p["ground_truth"] for p in test_pairs]
    noise_rates = [p.get("noise_rate") for p in test_pairs]

    # Baseline 1: raw OCR
    logger.info("Evaluating baseline (raw OCR)...")
    baseline_block = _metric_block(ocr_texts, gt_texts)

    # Baseline 2: spellchecker
    logger.info("Evaluating spellchecker baseline...")
    spell_texts = spellcheck_correct(ocr_texts)
    spell_block = _metric_block(spell_texts, gt_texts)

    # Model corrections
    logger.info("Loading model and generating corrections...")
    model, tokenizer = load_model(model_path)
    corrected_texts = correct_batch(ocr_texts, model, tokenizer, batch_size=16)
    corrected_block = _metric_block(corrected_texts, gt_texts)

    overall = {
        "num_samples": len(test_pairs),
        "raw_ocr":     baseline_block,
        "spellcheck":  spell_block,
        "model":       corrected_block,
        "improvement_vs_raw": {
            "cer":  calculate_improvement(baseline_block["cer"],  corrected_block["cer"]),
            "wer":  calculate_improvement(baseline_block["wer"],  corrected_block["wer"]),
        },
        "improvement_vs_spellcheck": {
            "cer":  calculate_improvement(spell_block["cer"],  corrected_block["cer"]),
            "wer":  calculate_improvement(spell_block["wer"],  corrected_block["wer"]),
        },
    }

    # Per noise level
    per_level = {}
    if any(r is not None for r in noise_rates):
        buckets = defaultdict(list)
        for i, r in enumerate(noise_rates):
            if r is not None:
                buckets[r].append(i)
        for rate in sorted(buckets):
            idxs = buckets[rate]
            ocr_s   = [ocr_texts[i]       for i in idxs]
            gt_s    = [gt_texts[i]        for i in idxs]
            spell_s = [spell_texts[i]     for i in idxs]
            corr_s  = [corrected_texts[i] for i in idxs]
            per_level[f"r={rate:.2f}"] = {
                "n":           len(idxs),
                "raw_ocr":     _metric_block(ocr_s,   gt_s),
                "spellcheck":  _metric_block(spell_s, gt_s),
                "model":       _metric_block(corr_s,  gt_s),
            }

    return overall, per_level, ocr_texts, gt_texts, spell_texts, corrected_texts


def plot_cer_wer_bar(overall: dict, out_dir: Path) -> None:
    metrics = ["CER", "WER"]
    raw   = [overall["raw_ocr"]["cer"],    overall["raw_ocr"]["wer"]]
    spell = [overall["spellcheck"]["cer"], overall["spellcheck"]["wer"]]
    model = [overall["model"]["cer"],      overall["model"]["wer"]]

    x = list(range(len(metrics)))
    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.27
    # Hatching keeps the series distinguishable when printed in black and white
    ax.bar([i - w for i in x], raw,   width=w, label="Noisy input",    color="#e74c3c",
           hatch="//", edgecolor="black", linewidth=0.8)
    ax.bar(x,                  spell, width=w, label="Spellchecker",   color="#f39c12",
           hatch="..", edgecolor="black", linewidth=0.8)
    ax.bar([i + w for i in x], model, width=w, label="Ours (flan-t5)", color="#2ecc71",
           hatch="\\\\", edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=13)
    ax.set_ylabel("Error rate (lower is better)")
    ax.set_title("CER / WER comparison across methods")
    ax.legend()
    ax.set_ylim(0, max(raw + spell + model) * 1.25)
    for i, vals in enumerate(zip(raw, spell, model)):
        for j, v in enumerate(vals):
            ax.text(i + (j - 1) * w, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    out = out_dir / "cer_wer_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Saved: {out}")


def plot_per_noise_level(per_level: dict, out_dir: Path) -> None:
    if not per_level:
        return
    rates = list(per_level.keys())
    raw_cer   = [per_level[r]["raw_ocr"]["cer"]    for r in rates]
    spell_cer = [per_level[r]["spellcheck"]["cer"] for r in rates]
    model_cer = [per_level[r]["model"]["cer"]      for r in rates]
    raw_wer   = [per_level[r]["raw_ocr"]["wer"]    for r in rates]
    spell_wer = [per_level[r]["spellcheck"]["wer"] for r in rates]
    model_wer = [per_level[r]["model"]["wer"]      for r in rates]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, raw, spell, model, title in [
        (axes[0], raw_cer, spell_cer, model_cer, "CER vs noise level"),
        (axes[1], raw_wer, spell_wer, model_wer, "WER vs noise level"),
    ]:
        ax.plot(rates, raw,   marker="o", label="Noisy input",    color="#e74c3c",
                linewidth=2, linestyle="-")
        ax.plot(rates, spell, marker="s", label="Spellchecker",   color="#f39c12",
                linewidth=2, linestyle="--")
        ax.plot(rates, model, marker="^", label="Ours (flan-t5)", color="#2ecc71",
                linewidth=2, linestyle=":")
        ax.set_xlabel("Noise rate")
        ax.set_ylabel("Error rate")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    out = out_dir / "metrics_per_noise_level.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Saved: {out}")


def plot_cer_distribution(ocr_texts, gt_texts, corrected_texts, out_dir: Path) -> None:
    baseline_cer_per  = [compute_cer(gt, ocr)  for ocr, gt in zip(ocr_texts, gt_texts)]
    corrected_cer_per = [compute_cer(gt, corr) for corr, gt in zip(corrected_texts, gt_texts)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    axes[0].hist(baseline_cer_per,  bins=20, color="#e74c3c", edgecolor="white")
    axes[0].set_title("CER distribution — Noisy input")
    axes[0].set_xlabel("CER")
    axes[0].set_ylabel("Number of samples")
    axes[1].hist(corrected_cer_per, bins=20, color="#2ecc71", edgecolor="white")
    axes[1].set_title("CER distribution — Corrected model")
    axes[1].set_xlabel("CER")
    plt.suptitle("Per-sample CER distribution (synthetic test set)", fontsize=13)
    plt.tight_layout()
    out = out_dir / "cer_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Saved: {out}")


def save_qualitative_examples(test_pairs, ocr_texts, corrected_texts, spell_texts, out_dir: Path, n: int = 10) -> None:
    gt_texts = [p["ground_truth"] for p in test_pairs]
    sorted_indices = sorted(range(len(test_pairs)), key=lambda i: len(gt_texts[i]), reverse=True)
    indices = sorted_indices[:n]
    examples = []
    for idx in indices:
        examples.append({
            "noise_rate":   test_pairs[idx].get("noise_rate"),
            "ocr":          ocr_texts[idx],
            "spellcheck":   spell_texts[idx],
            "corrected":    corrected_texts[idx],
            "ground_truth": gt_texts[idx],
        })
    out = out_dir / "qualitative_examples.json"
    with open(out, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {out}")


def run(model_path: str, data_path: str) -> None:
    out_dir = Path(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training curves
    history_path = out_dir / "train_history.json"
    if history_path.exists():
        plot_training_curves(history_path, out_dir)
    else:
        logger.warning(f"train_history.json not found at {history_path}, skipping curves.")

    # 2. Full evaluation: raw OCR, spellchecker, model — overall + per-noise-level
    test_pairs = json.load(open(data_path))
    overall, per_level, ocr_texts, gt_texts, spell_texts, corrected_texts = evaluate_all(
        test_pairs, model_path
    )

    results = {"overall": overall, "per_noise_level": per_level}
    results_path = out_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {results_path}")

    # 3. Plots
    plot_cer_wer_bar(overall, out_dir)
    plot_per_noise_level(per_level, out_dir)
    plot_cer_distribution(ocr_texts, gt_texts, corrected_texts, out_dir)
    save_qualitative_examples(test_pairs, ocr_texts, corrected_texts, spell_texts, out_dir, n=10)

    # 4. Save full predictions for downstream analysis (error categorization, etc.)
    predictions = [
        {
            "noise_rate":   test_pairs[i].get("noise_rate"),
            "ocr":          ocr_texts[i],
            "spellcheck":   spell_texts[i],
            "corrected":    corrected_texts[i],
            "ground_truth": gt_texts[i],
        }
        for i in range(len(test_pairs))
    ]
    preds_path = out_dir / "predictions.json"
    with open(preds_path, "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {preds_path}")

    # 5. Error analysis (categorized residual errors)
    try:
        from src.model.error_analysis import run_error_analysis
        run_error_analysis(predictions, out_dir)
    except Exception as e:
        logger.warning(f"Error analysis failed: {e}")

    logger.info("All post-processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ocr-corrector")
    parser.add_argument("--data",  type=str, default="data/pairs/synthetic_test.json")
    args = parser.parse_args()
    run(model_path=args.model, data_path=args.data)
