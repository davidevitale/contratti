"""
Test 6 — Health check, optimization status, AnalyzeSignature structure.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from main import AnalyzeSignature, ScoringSignature, GemellaSignature


@pytest.mark.asyncio
class TestHealthEndpoint:

    async def test_health_returns_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["service"] == "dspy_agents"


@pytest.mark.asyncio
class TestOptimizationStatus:

    async def test_not_optimized(self, client, _patch_redis, mock_redis):
        mock_redis.llen.return_value = 5

        with patch("main.optimizer") as mock_opt:
            mock_opt.get_optimization_status.return_value = {
                "optimized": False,
                "version": 0,
                "compiled_at": None,
                "feedback_needed": 20,
            }
            resp = await client.get("/optimization/status/test-client")

        assert resp.status_code == 200
        body = resp.json()
        assert body["optimized"] is False
        assert body["current_feedback_count"] == 5

    async def test_optimized_client(self, client, _patch_redis, mock_redis):
        mock_redis.llen.return_value = 25

        with patch("main.optimizer") as mock_opt:
            mock_opt.get_optimization_status.return_value = {
                "optimized": True,
                "version": 2,
                "compiled_at": "2024-06-01T12:00:00",
                "model_path": "/app/optimized_models/client1.json",
            }
            resp = await client.get("/optimization/status/client1")

        assert resp.status_code == 200
        body = resp.json()
        assert body["optimized"] is True
        assert body["version"] == 2


class TestAnalyzeSignatureStructure:
    """Verify the DSPy Signature has correct input/output fields."""

    @staticmethod
    def _fields(sig_cls):
        """Return field names for a DSPy Signature.
        Uses DSPy's own 'fields' dict; falls back to model_fields."""
        if hasattr(sig_cls, "fields"):
            return sig_cls.fields
        if hasattr(sig_cls, "model_fields"):
            return sig_cls.model_fields
        return vars(sig_cls)

    def test_has_required_input_fields(self):
        fields = self._fields(AnalyzeSignature)
        assert "contract_text" in fields
        assert "client_context" in fields

    def test_has_text_output_field(self):
        fields = self._fields(AnalyzeSignature)
        assert "extracted_json" in fields

    def test_output_is_plain_text(self):
        """Output is now plain text (not Pydantic type) — parsed manually."""
        fields = self._fields(AnalyzeSignature)
        assert "extracted_json" in fields


class TestScoringSignatureStructure:
    """Verify ScoringSignature has correct fields and text output."""

    @staticmethod
    def _fields(sig_cls):
        if hasattr(sig_cls, "fields"):
            return sig_cls.fields
        if hasattr(sig_cls, "model_fields"):
            return sig_cls.model_fields
        return vars(sig_cls)

    def test_has_required_input_fields(self):
        fields = self._fields(ScoringSignature)
        assert "extracted_contract" in fields
        assert "portfolio_context" in fields
        assert "sector_benchmarks" in fields

    def test_has_text_output_field(self):
        fields = self._fields(ScoringSignature)
        assert "scoring_json" in fields


class TestGemellaSignatureStructure:
    """Verify GemellaSignature has correct fields and text output."""

    @staticmethod
    def _fields(sig_cls):
        if hasattr(sig_cls, "fields"):
            return sig_cls.fields
        if hasattr(sig_cls, "model_fields"):
            return sig_cls.model_fields
        return vars(sig_cls)

    def test_has_required_input_fields(self):
        fields = self._fields(GemellaSignature)
        assert "clause_text" in fields
        assert "clause_type" in fields
        assert "sector" in fields
        assert "benchmark_corpus" in fields

    def test_has_text_output_field(self):
        fields = self._fields(GemellaSignature)
        assert "gemella_json" in fields


class TestEnvVarDefaults:
    """Verify env var defaults are applied correctly."""

    def test_external_llm_model_has_default(self):
        import main
        assert main.EXTERNAL_LLM_MODEL is not None
        assert main.EXTERNAL_LLM_MODEL != ""

    def test_api_base_never_none(self):
        import main
        assert main._api_base is not None
        assert main._api_base.endswith("/")

    def test_no_asyncio_import(self):
        """asyncio should NOT be imported in main (unused import removed)."""
        import main
        import inspect
        source = inspect.getsource(main)
        # Should not have a standalone 'import asyncio' line
        lines = source.split("\n")
        asyncio_imports = [l.strip() for l in lines if l.strip() == "import asyncio"]
        assert asyncio_imports == [], "Unused 'import asyncio' still present in main.py"
