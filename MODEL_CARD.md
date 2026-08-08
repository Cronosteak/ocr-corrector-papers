---
license: mit
language:
  - en
base_model: google/flan-t5-base
library_name: transformers
tags:
  - text2text-generation
  - ocr
  - ocr-post-correction
  - text-correction
  - seq2seq
  - flan-t5
  - scientific-text
metrics:
  - cer
  - wer
  - bleu
---

# OCR Corrector — flan-t5-base

Fine-tuned [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) (250M
parameters) that corrects OCR errors in electrical engineering academic text.

This is the model released with the paper **"Automatic OCR Error Correction in
Electrical Engineering Academic Papers Using Sequence-to-Sequence Models"**
(IEEE LA-CCI 2026).

- **Code and dataset:** https://github.com/Cronosteak/ocr-corrector-papers
- **License:** MIT

## Usage

The model expects the prefix `correct OCR: ` on every input.

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "Cronosteak/ocr-corrector-flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

text = "correct OCR: Thc powcr systcm opcrates at 60 Hz"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training data

6,440 synthetic OCR–ground-truth pairs built without manual annotation:
open-access electrical engineering abstracts were collected from the
[OpenAlex](https://openalex.org) API, then corrupted with synthetic OCR noise
(character confusions, spurious spaces/hyphens, deleted spaces, duplicated
characters) at four rates: r ∈ {0.02, 0.04, 0.06, 0.08}.

Splits are per-document (80/10/10) to avoid leakage between train and test.

## Results

Held-out test set of 676 pairs:

| Method | CER (%) ↓ | WER (%) ↓ | BLEU ↑ | EM (%) ↑ |
|---|---|---|---|---|
| Noisy input (no correction) | 3.28 | 21.99 | 63.95 | 0.00 |
| Spellchecker baseline | 3.35 | 15.88 | 73.79 | 5.47 |
| **This model** | **2.05** | **4.21** | **91.94** | **44.23** |

A 37.5% relative CER reduction and an 80.9% relative WER reduction over the
uncorrected input. Per-noise-rate metrics are in `eval_results.json`.

Ablations showed that **training across multiple noise levels is the single most
important factor for generalization**: restricting training to r=0.04 alone drops
exact match from 44.23% to 35.21%.

## Training procedure

| Hyperparameter | Value |
|---|---|
| Base model | google/flan-t5-base |
| Optimizer | Adafactor |
| Learning rate | 1e-4 |
| Batch size | 8 (grad. accumulation 2) |
| Epochs | 15 (early stopping, patience 3) |
| Warmup steps | 200 |
| Weight decay | 0.01 |
| Max input/output length | 256 / 256 |
| Seed | 42 |

## Limitations

- Trained on **synthetic** OCR noise, not on real Tesseract output. Real-world
  error distributions may differ.
- Domain is electrical engineering and adjacent fields (computer science,
  engineering) in English; performance on other domains or languages is untested.
- Residual errors concentrate on lowercase common words at the highest noise rate
  (r=0.08), where corrupted word shapes can be decoded into plausible but wrong
  alternatives. Because it is a generative model, it can rewrite text that was
  already correct — review output before using it in a critical pipeline.
- Inputs are truncated at 256 tokens; long passages must be chunked.

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
