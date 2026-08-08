"""
text_cleaner.py — Text normalization and cleaning functions.
"""

import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to their NFC form."""
    return unicodedata.normalize("NFC", text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse repeated whitespace into a single space and strip the edges."""
    return re.sub(r"\s+", " ", text).strip()


def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """Remove special characters, optionally keeping basic punctuation."""
    if keep_punctuation:
        return re.sub(r"[^\w\s.,;:!?¿¡()\"'-]", "", text)
    return re.sub(r"[^\w\s]", "", text)


def clean_ocr_artifacts(text: str) -> str:
    """Drop common OCR artifacts: stray characters, symbol-only lines."""
    # Drop lines that are symbols only or very short (< 3 chars)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 3:
            continue
        if re.match(r"^[\W_]+$", stripped):
            continue
        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def full_clean(text: str) -> str:
    """Apply all cleaning steps in order."""
    text = normalize_unicode(text)
    text = clean_ocr_artifacts(text)
    text = remove_extra_whitespace(text)
    return text
