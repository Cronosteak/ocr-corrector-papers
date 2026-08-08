"""
predict.py — Uses the trained model to correct noisy OCR text.
"""

import logging
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

_PREFIX = "correct OCR: "
_MAX_INPUT = 512
_MAX_OUTPUT = 256


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer


def correct_text(text: str, model=None, tokenizer=None, model_path: str | None = None) -> str:
    """
    Correct a single noisy OCR text. Either pass a loaded model/tokenizer or a
    model_path to load them from.
    """
    if model is None or tokenizer is None:
        if model_path is None:
            raise ValueError("Provide either model/tokenizer or model_path")
        model, tokenizer = load_model(model_path)

    device = next(model.parameters()).device
    input_text = _PREFIX + text
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=_MAX_INPUT,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=_MAX_OUTPUT, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def correct_batch(
    texts: list[str],
    model,
    tokenizer,
    batch_size: int = 16,
) -> list[str]:
    device = next(model.parameters()).device
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = [_PREFIX + t for t in texts[i : i + batch_size]]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=_MAX_INPUT,
            truncation=True,
            padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=_MAX_OUTPUT, num_beams=4)
        results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Correct OCR text")
    parser.add_argument("--input", type=str, required=True, help="OCR text to correct")
    parser.add_argument("--model", type=str, default="models/ocr-corrector", help="Path to the model")
    args = parser.parse_args()

    corrected = correct_text(args.input, model_path=args.model)
    print(f"Original:  {args.input}")
    print(f"Corrected: {corrected}")
