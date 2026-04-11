"""
test_metrics.py — Tests para las funciones de métricas.
"""

import pytest

from src.utils.metrics import calculate_improvement
from src.utils.text_cleaner import (
    normalize_unicode,
    remove_extra_whitespace,
    clean_ocr_artifacts,
    full_clean,
)


class TestTextCleaner:
    def test_normalize_unicode(self):
        # Carácter compuesto vs precompuesto
        text = "café"
        result = normalize_unicode(text)
        assert isinstance(result, str)

    def test_remove_extra_whitespace(self):
        assert remove_extra_whitespace("  hello   world  ") == "hello world"
        assert remove_extra_whitespace("a\t\nb") == "a b"

    def test_clean_ocr_artifacts(self):
        text = "Valid line here\n--\na\n====\nAnother valid line"
        result = clean_ocr_artifacts(text)
        assert "Valid line here" in result
        assert "Another valid line" in result
        assert "====" not in result

    def test_full_clean(self):
        text = "  Hello   World  \n--\n  Valid text here  "
        result = full_clean(text)
        assert "Hello" in result
        assert "Valid text here" in result


class TestMetrics:
    def test_calculate_improvement(self):
        result = calculate_improvement(0.5, 0.2)
        assert result["baseline"] == 0.5
        assert result["corrected"] == 0.2
        assert result["absolute_improvement"] == 0.3
        assert result["relative_improvement_pct"] == 60.0

    def test_calculate_improvement_zero_baseline(self):
        result = calculate_improvement(0.0, 0.0)
        assert result["relative_improvement_pct"] == 0.0

    def test_calculate_improvement_no_change(self):
        result = calculate_improvement(0.4, 0.4)
        assert result["absolute_improvement"] == 0.0
        assert result["relative_improvement_pct"] == 0.0
