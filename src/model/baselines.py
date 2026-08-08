"""
baselines.py — Non-neural baselines for comparison.
Currently: dictionary-based correction (pyspellchecker).
"""

import logging
import re

logger = logging.getLogger(__name__)


def _match_case(source: str, target: str) -> str:
    """Apply the capitalization of `source` to `target`."""
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def spellcheck_correct(texts: list[str]) -> list[str]:
    """
    Correct OCR text word by word with a spellchecker. Only alphabetic tokens
    are corrected; punctuation, numbers, symbols and casing are preserved.
    """
    try:
        from spellchecker import SpellChecker
    except ImportError as e:
        raise ImportError(
            "pyspellchecker is not installed. Add `pyspellchecker` to requirements.txt"
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
                # Leave all-caps acronyms (NDVI, UAS, ...) untouched
                corrected_tokens.append(tok)
        out.append("".join(corrected_tokens))
    return out
