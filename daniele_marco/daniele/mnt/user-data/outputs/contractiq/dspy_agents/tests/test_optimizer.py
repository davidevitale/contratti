"""
Test 8 — PerClientOptimizer and user_satisfaction_metric.
No LLM or filesystem needed (uses tmp_path).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import os
import sys

AGENTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AGENTS_ROOT not in sys.path:
    sys.path.insert(0, AGENTS_ROOT)

# Patch dspy.OpenAI at import time
with patch("dspy.OpenAI", return_value=MagicMock()), \
     patch("dspy.settings.configure"):
    pass

from feedback.optimizer import (
    PerClientOptimizer,
    user_satisfaction_metric,
    FEEDBACK_THRESHOLD,
)


# ────────────────────────────────────────────────────────────────────────────
# Metric tests
# ────────────────────────────────────────────────────────────────────────────

class TestUserSatisfactionMetric:

    @staticmethod
    def _example(rating, comment=""):
        ex = MagicMock()
        ex.user_rating = rating
        ex.user_comment = comment
        return ex

    @staticmethod
    def _prediction(answer="answer", sources="[]"):
        p = MagicMock()
        p.answer = answer
        p.sources = sources
        return p

    def test_five_stars_gives_max(self):
        score = user_satisfaction_metric(self._example(5), self._prediction())
        assert score == 1.0

    def test_one_star_gives_zero(self):
        score = user_satisfaction_metric(self._example(1), self._prediction())
        assert score == 0.0

    def test_three_stars_gives_half(self):
        score = user_satisfaction_metric(self._example(3), self._prediction())
        assert score == 0.5

    def test_citation_bonus(self):
        """Answers with sources get +0.1 bonus."""
        sources = json.dumps([{"contract": "C1", "clause": "Art 1"}])
        score = user_satisfaction_metric(
            self._example(3),
            self._prediction(sources=sources),
        )
        assert score == 0.6  # 0.5 + 0.1

    def test_positive_comment_bonus(self):
        score = user_satisfaction_metric(
            self._example(4, "Ottimo risultato!"),
            self._prediction(),
        )
        assert abs(score - 0.85) < 1e-9  # 0.8 + 0.05

    def test_negative_comment_penalty(self):
        score = user_satisfaction_metric(
            self._example(4, "Risposta sbagliato completamente"),
            self._prediction(),
        )
        assert score == 0.65  # 0.8 - 0.15

    def test_score_clamped_to_one(self):
        """5 stars + citations + positive comment must not exceed 1.0."""
        sources = json.dumps([{"contract": "C1"}])
        score = user_satisfaction_metric(
            self._example(5, "Perfetto"),
            self._prediction(sources=sources),
        )
        assert score == 1.0


# ────────────────────────────────────────────────────────────────────────────
# Optimizer lifecycle
# ────────────────────────────────────────────────────────────────────────────

class TestPerClientOptimizer:

    def test_no_optimized_model_initially(self, tmp_path):
        with patch("feedback.optimizer.OPTIMIZED_MODELS_PATH", tmp_path):
            opt = PerClientOptimizer()
            assert opt.has_optimized_model("new_client") is False

    def test_get_optimization_status_unoptimized(self, tmp_path):
        with patch("feedback.optimizer.OPTIMIZED_MODELS_PATH", tmp_path):
            opt = PerClientOptimizer()
            status = opt.get_optimization_status("new_client")
            assert status["optimized"] is False
            assert status["version"] == 0

    def test_load_falls_back_to_base(self, tmp_path):
        with patch("feedback.optimizer.OPTIMIZED_MODELS_PATH", tmp_path):
            opt = PerClientOptimizer()
            base = MagicMock(name="base_orchestrator")
            result = opt.load_optimized_orchestrator("no_model", base)
            assert result is base

    def test_below_threshold_returns_none(self, tmp_path):
        with patch("feedback.optimizer.OPTIMIZED_MODELS_PATH", tmp_path):
            opt = PerClientOptimizer()
            base = MagicMock()
            examples = [MagicMock() for _ in range(5)]
            result = opt.run_optimization("c1", base, examples)
            assert result is None

    def test_threshold_constant(self):
        assert FEEDBACK_THRESHOLD == 20
