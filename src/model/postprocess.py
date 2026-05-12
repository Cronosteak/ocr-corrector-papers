"""
postprocess.py — Post-training analysis: training curves, CER/WER bar chart,
CER per-sample histogram, and qualitative examples.
All outputs are saved as PNG/JSON files (no display needed — cluster-safe).
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required on cluster
import matplotlib.pyplot as plt
from jiwer import cer as compute_cer

from src.model.evaluate import evaluate_model
from src.model.predict import correct_batch, correct_text, load_model

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


def plot_cer_wer_bar(results: dict, out_dir: Path) -> None:
    metrics = ["CER", "WER"]
    baseline  = [results["baseline_cer"], results["baseline_wer"]]
    corrected = [results["cer"]["corrected"], results["wer"]["corrected"]]
    improvement = [results["cer"]["relative_improvement_pct"], results["wer"]["relative_improvement_pct"]]

    x = range(len(metrics))
    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar([i - 0.2 for i in x], baseline,  width=0.4, label="Baseline (raw OCR)", color="#e74c3c")
    bars2 = ax.bar([i + 0.2 for i in x], corrected, width=0.4, label="Corrected (flan-t5)", color="#2ecc71")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, fontsize=13)
    ax.set_ylabel("Error rate (lower is better)")
    ax.set_title("CER / WER: Baseline vs Corrected model (synthetic test set)")
    ax.legend()
    ax.set_ylim(0, 1.1)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", fontsize=10)
    for i, pct in enumerate(improvement):
        color = "#27ae60" if pct > 0 else "#c0392b"
        ax.text(i, 1.07, f"{'+' if pct > 0 else ''}{pct:.1f}%", ha="center", fontsize=9, color=color)
    plt.tight_layout()
    out = out_dir / "cer_wer_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Saved: {out}")


def plot_cer_distribution(ocr_texts, gt_texts, corrected_texts, out_dir: Path) -> None:
    baseline_cer_per  = [compute_cer(gt, ocr)  for ocr, gt in zip(ocr_texts, gt_texts)]
    corrected_cer_per = [compute_cer(gt, corr) for corr, gt in zip(corrected_texts, gt_texts)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    axes[0].hist(baseline_cer_per,  bins=20, color="#e74c3c", edgecolor="white")
    axes[0].set_title("CER distribution — Baseline (raw OCR)")
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


def save_qualitative_examples(test_pairs, ocr_texts, corrected_texts, out_dir: Path, n: int = 5) -> None:
    gt_texts = [p["ground_truth"] for p in test_pairs]
    # Seleccionar los n ejemplos con oraciones más largas (más informativos)
    sorted_indices = sorted(range(len(test_pairs)), key=lambda i: len(gt_texts[i]), reverse=True)
    indices = sorted_indices[:n]
    examples = []
    for idx in indices:
        examples.append({
            "ocr":        ocr_texts[idx],
            "corrected":  corrected_texts[idx],
            "ground_truth": gt_texts[idx],
        })
        logger.info(f"\n--- Example {idx} ---\n"
                    f"OCR:       {ocr_texts[idx][:120]}\n"
                    f"Corrected: {corrected_texts[idx][:120]}\n"
                    f"GT:        {gt_texts[idx][:120]}")
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

    # 2. Evaluate CER/WER
    logger.info("Running evaluation...")
    results = evaluate_model(pairs_path=data_path, model_path=model_path)
    results_path = out_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {results_path}")

    if "cer" not in results:
        logger.error("Evaluation did not return model metrics. Aborting plots.")
        return

    plot_cer_wer_bar(results, out_dir)

    # 3. Per-sample CER histogram + qualitative examples (reuse predictions)
    logger.info("Loading model for per-sample analysis...")
    model, tokenizer = load_model(model_path)
    test_pairs = json.load(open(data_path))
    ocr_texts = [p["ocr"] for p in test_pairs]
    gt_texts  = [p["ground_truth"] for p in test_pairs]
    corrected_texts = correct_batch(ocr_texts, model, tokenizer, batch_size=16)

    plot_cer_distribution(ocr_texts, gt_texts, corrected_texts, out_dir)
    save_qualitative_examples(test_pairs, ocr_texts, corrected_texts, out_dir, n=10)

    logger.info("All post-processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ocr-corrector")
    parser.add_argument("--data",  type=str, default="data/pairs/synthetic_test.json")
    args = parser.parse_args()
    run(model_path=args.model, data_path=args.data)
