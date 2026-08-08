# OCR Corrector Papers

Automatic OCR error correction for electrical engineering academic papers, using
sequence-to-sequence models fine-tuned on data built automatically from
[OpenAlex](https://openalex.org).

This is the official code release for the paper:

> **Automatic OCR Error Correction in Electrical Engineering Academic Papers Using Sequence-to-Sequence Models**
> Breinner Farid Espinosa Ortiz, Leonardo Alfredo Forero Mendoza, Marco Aurélio C. Pacheco, Evelyn C. S. Batista
> *2026 IEEE Latin American Conference on Computational Intelligence (LA-CCI)*

**Model weights:** [`Cronosteak/ocr-corrector-flan-t5-base`](https://huggingface.co/Cronosteak/ocr-corrector-flan-t5-base) on the Hugging Face Hub.

---

## Results

Fine-tuned `google/flan-t5-base` (250M parameters) on 6,440 synthetic
OCR–ground-truth pairs, evaluated on a held-out 676-pair test set.

| Method | CER (%) ↓ | WER (%) ↓ | BLEU ↑ | EM (%) ↑ |
|---|---|---|---|---|
| Noisy input (no correction) | 3.28 | 21.99 | 63.95 | 0.00 |
| Spellchecker baseline | 3.35 | 15.88 | 73.79 | 5.47 |
| **flan-t5-base (ours)** | **2.05** | **4.21** | **91.94** | **44.23** |

That is a **37.5% relative CER reduction** and an **80.9% relative WER
reduction** over the uncorrected input.

Note that the dictionary-based spellchecker *fails to improve CER* over doing
nothing: it introduces spurious character-level edits on domain terms and proper
nouns that the noisy input had preserved correctly.

### Ablation study

| Variant | CER (%) | WER (%) | BLEU | EM (%) |
|---|---|---|---|---|
| Full model (ours) | 2.05 | 4.21 | 91.94 | **44.23** |
| w/o warmup | 2.09 | 4.41 | 91.74 | 40.53 |
| w/o weight decay | **1.92** | **4.15** | **92.19** | 41.72 |
| Single noise (r=0.04 only) | 2.40 | 5.33 | 89.89 | 35.21 |

**Training across multiple noise levels is the single most important factor for
generalization** — restricting training to one noise rate drops exact match by
9 percentage points.

All raw numbers above are reproducible from the JSON files in [`results/`](results/).

---

## Repository structure

```
ocr-corrector-papers/
├── src/
│   ├── pipeline/           # Data pipeline
│   │   ├── fetch_openalex.py    # Query OpenAlex for open-access papers
│   │   ├── download_pdfs.py     # Download PDFs
│   │   ├── ocr_extract.py       # Tesseract OCR extraction
│   │   ├── align_text.py        # Align noisy OCR with ground truth
│   │   └── build_dataset.py     # Synthetic noise injection + train/val/test splits
│   ├── model/
│   │   ├── train.py             # Fine-tuning loop
│   │   ├── postprocess.py       # Full evaluation: metrics, baselines, plots
│   │   ├── evaluate.py          # Lightweight metrics-only entry point
│   │   ├── baselines.py         # Spellchecker baseline (used by postprocess)
│   │   ├── error_analysis.py    # Token-level error categorization
│   │   ├── compare_ablations.py # Ablation comparison tables
│   │   └── predict.py           # Inference on new text
│   ├── utils/                   # Metrics, logging, text cleaning, timing
│   └── api/                     # Optional FastAPI REST service
├── configs/                # Training and ablation YAML configs
├── data/pairs/             # Pre-generated 6,440-pair dataset (versioned)
├── results/                # Evaluation artifacts for every trained variant
├── slurm_scripts/          # SLURM job scripts used to produce the results
├── notebooks/              # Exploration and results notebooks
├── tests/                  # Unit tests
└── docker/                 # Dockerfile for reproducible runs
```

Heavy artifacts (raw PDFs, OCR text, model checkpoints) are **not** versioned.
The dataset needed to reproduce every number in the paper *is* — see
[`data/pairs/`](data/pairs/).

---

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env    # then edit the paths/credentials
```

System dependencies for the OCR stage: `tesseract-ocr` and `poppler-utils`.

```bash
sudo apt install tesseract-ocr poppler-utils
```

A [Dockerfile](docker/Dockerfile) is provided if you prefer a pinned environment.

---

## Reproducing the paper

The dataset is already committed, so you can skip straight to training:

```bash
# 1. Train the model (config: configs/train_config.yaml)
python -m src.model.train

# 2. Full evaluation: CER/WER/BLEU/EM, per-noise-rate breakdown, both
#    baselines (noisy input + spellchecker), plots and qualitative examples.
#    Writes eval_results.json, predictions.json and the paper figures.
python -m src.model.postprocess --model models/ocr-corrector \
                                --data data/pairs/synthetic_test.json

# 3. Token-level error categorization (Figure 5 in the paper)
python -m src.model.error_analysis --predictions models/ocr-corrector/predictions.json
```

`src/model/evaluate.py` is a lighter standalone entry point if you only want the
headline metrics printed as JSON:

```bash
python -m src.model.evaluate --model models/ocr-corrector
```

### Ablations

`slurm_scripts/run_ablations.sh` submits the three variants via **SLURM**
(`sbatch`), which is how they were run for the paper. Edit `REPO_PATH` in the
`.srm` files to point at your checkout first.

```bash
./slurm_scripts/run_ablations.sh          # SLURM clusters
python -m src.model.compare_ablations     # -> models/ablation_results.md
```

Without SLURM, train each variant directly and then compare:

```bash
python -m src.model.train --config configs/ablation_no_warmup.yaml
python -m src.model.train --config configs/ablation_no_wd.yaml
python -m src.model.train --config configs/ablation_single_noise.yaml
python -m src.model.compare_ablations
```

To rebuild the dataset from scratch instead — note this re-downloads and re-OCRs
several hundred PDFs and takes roughly 3 hours, dominated by the OCR stage:

```bash
python -m src.pipeline.fetch_openalex    # OpenAlex query -> data/works.json
python -m src.pipeline.download_pdfs     # -> data/raw/*.pdf
python -m src.pipeline.ocr_extract       # -> data/ocr/*.txt
python -m src.pipeline.build_dataset     # -> data/pairs/*.json
```

Stage timings from our run are recorded in [`data/pipeline_stats.json`](data/pipeline_stats.json).

---

## Usage

Correct a piece of OCR text:

```bash
python -m src.model.predict --input "Thc powcr systcm opcrates at 60 Hz"
```

Or serve it over HTTP:

```bash
uvicorn src.api.server:app --reload
```

---

## Configuration

| File | Purpose |
|---|---|
| `configs/openalex_query.yaml` | OpenAlex search filters (field, year range, OA status) |
| `configs/train_config.yaml` | Base model, hyperparameters, splits |
| `configs/ablation_no_warmup.yaml` | Ablation: warmup steps set to 0 |
| `configs/ablation_no_wd.yaml` | Ablation: weight decay set to 0 |
| `configs/ablation_single_noise.yaml` | Ablation: single noise rate (r=0.04) |

---

## Tests

```bash
pytest tests/
```

---

## Citation

```bibtex
@inproceedings{espinosa2026ocr,
  title     = {Automatic OCR Error Correction in Electrical Engineering
               Academic Papers Using Sequence-to-Sequence Models},
  author    = {Espinosa Ortiz, Breinner Farid and
               Forero Mendoza, Leonardo Alfredo and
               Pacheco, Marco Aur{\'e}lio C. and
               Batista, Evelyn C. S.},
  booktitle = {2026 IEEE Latin American Conference on Computational
               Intelligence (LA-CCI)},
  year      = {2026},
  publisher = {IEEE}
}
```

## License

Released under the [MIT License](LICENSE).
