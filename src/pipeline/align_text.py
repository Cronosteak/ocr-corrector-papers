"""
align_text.py — Aligns noisy OCR text with clean ground-truth text using edit
distance, producing training pairs.
"""

import logging
from pathlib import Path

from Levenshtein import ratio as levenshtein_ratio

logger = logging.getLogger(__name__)

# The abstract is one continuous block — split it into sentences on ". "
def _split_into_sentences(text: str) -> list[str]:
    sentences = []
    for sent in text.replace("\n", " ").split(". "):
        sent = sent.strip()
        if len(sent) > 30:  # ignore very short fragments
            sentences.append(sent)
    return sentences


def align_paragraphs(
    ocr_text: str, gt_text: str, min_similarity: float = 0.4
) -> list[dict]:
    """
    Match each ground-truth sentence to its most similar OCR chunk.

    The GT is the abstract (a continuous block) while the OCR is the whole
    document, so the full abstract is also tried as a single chunk.
    Returns pairs of {ocr, ground_truth, similarity}.
    """
    # Split the OCR into chunks on blank lines
    ocr_chunks = [p.replace("\n", " ").strip() for p in ocr_text.split("\n\n") if len(p.strip()) > 30]

    # Sliding window: also consider merged pairs of adjacent chunks
    combined_chunks = ocr_chunks.copy()
    for i in range(len(ocr_chunks) - 1):
        merged = ocr_chunks[i] + " " + ocr_chunks[i + 1]
        combined_chunks.append(merged)

    # Split the GT into sentences
    gt_sentences = _split_into_sentences(gt_text)

    # Also try the full abstract as a single GT chunk
    gt_full = gt_text.replace("\n", " ").strip()
    if len(gt_full) > 30:
        gt_sentences.append(gt_full)

    used_chunks = set()
    pairs = []

    for gt_sent in gt_sentences:
        best_chunk = None
        best_score = 0.0

        for i, chunk in enumerate(combined_chunks):
            score = levenshtein_ratio(gt_sent, chunk)
            if score > best_score:
                best_score = score
                best_chunk = (i, chunk)

        if best_chunk and best_score >= min_similarity and best_chunk[0] not in used_chunks:
            used_chunks.add(best_chunk[0])
            pairs.append({
                "ocr": best_chunk[1],
                "ground_truth": gt_sent,
                "similarity": round(best_score, 4),
            })

    return pairs


def align_files(ocr_path: Path, gt_path: Path, min_similarity: float = 0.4) -> list[dict]:
    """Align one OCR file with its corresponding ground-truth file."""
    ocr_text = ocr_path.read_text(encoding="utf-8")
    gt_text = gt_path.read_text(encoding="utf-8")
    return align_paragraphs(ocr_text, gt_text, min_similarity)


if __name__ == "__main__":
    import json

    ocr_dir = Path("data/ocr")
    gt_dir = Path("data/ground_truth")

    all_pairs = []
    for ocr_path in sorted(ocr_dir.glob("*.txt")):
        gt_path = gt_dir / ocr_path.name
        if not gt_path.exists():
            logger.warning(f"No ground truth for {ocr_path.name}, skipping.")
            continue

        pairs = align_files(ocr_path, gt_path)
        all_pairs.extend(pairs)
        print(f"{ocr_path.stem}: {len(pairs)} aligned pairs")

    output = Path("data/pairs/aligned_sample.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"\nTotal: {len(all_pairs)} pairs → {output}")
