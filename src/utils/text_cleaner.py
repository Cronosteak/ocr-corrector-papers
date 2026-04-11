"""
text_cleaner.py — Funciones de normalización y limpieza de texto.
"""

import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """Normaliza caracteres Unicode a su forma NFC."""
    return unicodedata.normalize("NFC", text)


def remove_extra_whitespace(text: str) -> str:
    """Reduce espacios múltiples a uno solo y elimina espacios al inicio/final."""
    return re.sub(r"\s+", " ", text).strip()


def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """
    Elimina caracteres especiales del texto.

    Args:
        text: Texto a limpiar.
        keep_punctuation: Si True, mantiene puntuación básica.

    Returns:
        Texto limpio.
    """
    if keep_punctuation:
        return re.sub(r"[^\w\s.,;:!?¿¡()\"'-]", "", text)
    return re.sub(r"[^\w\s]", "", text)


def clean_ocr_artifacts(text: str) -> str:
    """
    Limpia artefactos comunes del OCR.
    Ejemplos: caracteres sueltos, líneas de solo símbolos, etc.
    """
    # Eliminar líneas que son solo símbolos o muy cortas (< 3 chars)
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
    """Aplica todas las limpiezas en orden."""
    text = normalize_unicode(text)
    text = clean_ocr_artifacts(text)
    text = remove_extra_whitespace(text)
    return text
