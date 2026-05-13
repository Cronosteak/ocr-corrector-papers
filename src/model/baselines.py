"""
baselines.py — Baselines no-neuronales para comparación.
Actualmente: corrección por diccionario (pyspellchecker).
"""

import logging
import re

logger = logging.getLogger(__name__)


def _match_case(source: str, target: str) -> str:
    """Restaura la capitalización de `source` sobre `target`."""
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def spellcheck_correct(texts: list[str]) -> list[str]:
    """
    Corrige texto OCR aplicando un spellchecker palabra por palabra.
    Solo corrige tokens alfabéticos; preserva puntuación, números, símbolos
    y capitalización original.
    """
    try:
        from spellchecker import SpellChecker
    except ImportError as e:
        raise ImportError(
            "pyspellchecker no está instalado. Añade `pyspellchecker` a requirements.txt"
        ) from e

    sc = SpellChecker(distance=2)
    token_re = re.compile(r"(\w+|\W+)", re.UNICODE)

    out = []
    for text in texts:
        tokens = token_re.findall(text)
        corrected_tokens = []
        for tok in tokens:
            if tok.isalpha() and len(tok) > 1 and not tok.isupper():
                candidate = sc.correction(tok)
                if candidate:
                    corrected_tokens.append(_match_case(tok, candidate))
                else:
                    corrected_tokens.append(tok)
            else:
                # Preserva acrónimos en mayúsculas (NDVI, UAS, etc.) sin tocar
                corrected_tokens.append(tok)
        out.append("".join(corrected_tokens))
    return out
