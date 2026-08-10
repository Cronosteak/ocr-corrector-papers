import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RUNS = {
    "Full (ours)":      "models/ocr-corrector",
    "no warmup":        "models/ablation_no_warmup",
    "no weight decay":  "models/ablation_no_wd",
    "single noise (r=0.04)": "models/ablation_single_noise",
}


def _load(path: str) -> dict | None:
    p = Path(path) / "eval_results.json"
    if not p.exists():
        logger.warning(f"missing {p}")
        return None
    return json.load(open(p))


def build_table() -> str:
    rows = []
    header = "| Variant | CER (%) | WER (%) | BLEU | EM (%) |"
    sep    = "|---|---|---|---|---|"
    rows.append(header)
    rows.append(sep)
    for name, run_dir in RUNS.items():
        data = _load(run_dir)
        if data is None:
            rows.append(f"| {name} | -- | -- | -- | -- |")
            continue
        m = data["overall"]["model"]
        rows.append(
            f"| {name} | {100*m['cer']:.2f} | {100*m['wer']:.2f} | "
            f"{m['bleu']:.2f} | {m['exact_match_pct']:.2f} |"
        )
    return "\n".join(rows)


def build_per_noise_table() -> str:
    rows = ["| Variant | r=0.02 WER | r=0.04 WER | r=0.06 WER | r=0.08 WER |",
            "|---|---|---|---|---|"]
    for name, run_dir in RUNS.items():
        data = _load(run_dir)
        if data is None:
            rows.append(f"| {name} | -- | -- | -- | -- |")
            continue
        pn = data.get("per_noise_level", {})
        cells = []
        for r in ["r=0.02", "r=0.04", "r=0.06", "r=0.08"]:
            if r in pn:
                cells.append(f"{100*pn[r]['model']['wer']:.2f}")
            else:
                cells.append("--")
        rows.append(f"| {name} | {' | '.join(cells)} |")
    return "\n".join(rows)


if __name__ == "__main__":
    out_md = "# Ablation results\n\n## Overall (test set, 676 pairs)\n\n"
    out_md += build_table()
    out_md += "\n\n## Per-noise-rate WER (model only)\n\n"
    out_md += build_per_noise_table()
    out_md += "\n"
    Path("models").mkdir(exist_ok=True)
    out_path = Path("models/ablation_results.md")
    out_path.write_text(out_md)
    print(out_md)
    logger.info(f"Saved: {out_path}")
