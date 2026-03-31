"""
Test 3 — /analyze/contract endpoint.
Mocks dspy.Predict so no LLM call is made.
Validates the full extraction → scoring → gemella pipeline.
All three steps use dspy.Predict + manual JSON parsing.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from main import (
    ContractData, ProductService, PenaltyClause, RiskFlag,
    ScoringData, GemellaData,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_extraction_mock(contract_data: ContractData):
    """Create a mock Predict result with extracted_json field."""
    pred = MagicMock()
    pred.extracted_json = json.dumps(contract_data.model_dump())
    return pred


def _make_scoring_mock(scoring_data: ScoringData = None):
    sd = scoring_data or ScoringData(
        overall_score=75,
        revenue_quality_score=80,
        risk_exposure_score=60,
        strategic_value_score=70,
        concentration_risk_score=40,
    )
    pred = MagicMock()
    pred.scoring_json = json.dumps(sd.model_dump())
    return pred


def _make_gemella_mock(gemella_data: GemellaData = None):
    gd = gemella_data or GemellaData(
        percentile_ranking=65,
        benchmark_summary="Typical for sector.",
        deviation_analysis="Slightly above average liability cap.",
        negotiation_script="Propose reducing to 100% cap.",
        priority="medium",
    )
    pred = MagicMock()
    pred.gemella_json = json.dumps(gd.model_dump())
    return pred


def _build_predict_side_effect(extraction_cd, scoring_data=None, gemella_data=None):
    """
    Returns a side_effect function for dspy.Predict.
    Predict is called 3 times in order: AnalyzeSignature, ScoringSignature, GemellaSignature.
    Each call returns a mock instance, which when called returns the prediction.
    """
    call_count = [0]

    def _factory(signature_cls, **kwargs):
        mock_instance = MagicMock()
        idx = call_count[0]
        call_count[0] += 1

        if idx == 0:  # AnalyzeSignature
            mock_instance.return_value = _make_extraction_mock(extraction_cd)
        elif idx == 1:  # ScoringSignature
            mock_instance.return_value = _make_scoring_mock(scoring_data)
        else:  # GemellaSignature (idx == 2)
            mock_instance.return_value = _make_gemella_mock(gemella_data)
        return mock_instance

    return _factory


def _build_predict_no_gemella(extraction_cd, scoring_data=None):
    """For cases where gemella is not triggered (no risk_flags)."""
    call_count = [0]

    def _factory(signature_cls, **kwargs):
        mock_instance = MagicMock()
        idx = call_count[0]
        call_count[0] += 1

        if idx == 0:
            mock_instance.return_value = _make_extraction_mock(extraction_cd)
        else:
            mock_instance.return_value = _make_scoring_mock(scoring_data)
        return mock_instance

    return _factory


# ── Tests ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAnalyzeContractEndpoint:

    async def test_successful_extraction(self, client, sample_contract, _patch_redis):
        """Full pipeline returns valid extraction, scoring, gemella."""
        cd = ContractData(
            client_name="Acme Manufacturing S.p.A.",
            contract_type="SaaS",
            total_value_eur=210000.0,
            annual_recurring_revenue=70000.0,
            start_date="2024-03-01",
            end_date="2027-02-28",
            auto_renewal=True,
            renewal_notice_days=90,
            payment_terms_days=60,
            discount_percentage=8.0,
            sla_uptime_percentage=99.5,
            products_services=[
                ProductService(name="FactoryOS Core", value_eur=45000, recurring=True),
            ],
            penalty_clauses=[
                PenaltyClause(trigger="Late delivery", amount_eur=500, type="daily"),
            ],
            risk_flags=[
                RiskFlag(flag="Exclusive clause", severity="high", clause="Art. 6.2"),
            ],
            non_standard_terms=["Exclusive clause art. 6.2"],
        )

        with patch("main.dspy.Predict", side_effect=_build_predict_side_effect(cd)):
            resp = await client.post("/analyze/contract", json={
                "client_id": "test-client",
                "contract_text": sample_contract,
                "sector": "manufacturing",
            })

        assert resp.status_code == 200
        body = resp.json()
        analysis = body["analysis"]

        # Extraction checks
        ext = analysis["extraction"]
        assert ext["client_name"] == "Acme Manufacturing S.p.A."
        assert ext["total_value_eur"] == 210000.0
        assert ext["auto_renewal"] is True
        assert len(ext["products_services"]) == 1
        assert ext["risk_flags"][0]["severity"] == "high"

        # Scoring checks
        sc = analysis["scoring"]
        assert sc["overall_score"] == 75
        assert 0 <= sc["risk_exposure_score"] <= 100

        # Gemella checks
        gem = analysis["clausola_gemella"]
        assert len(gem) == 1
        assert gem[0]["percentile_ranking"] == 65
        assert gem[0]["priority"] == "medium"

    async def test_extraction_fallback_on_llm_failure(self, client, sample_contract, _patch_redis):
        """If Predict raises on extraction, endpoint returns defaults — no crash."""
        call_count = [0]

        def _factory(sig_cls, **kwargs):
            mock_instance = MagicMock()
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:  # extraction fails
                mock_instance.side_effect = ValueError("LLM returned garbage")
            else:  # scoring still works (extraction defaults have no risk_flags → no gemella)
                mock_instance.return_value = _make_scoring_mock()
            return mock_instance

        with patch("main.dspy.Predict", side_effect=_factory):
            resp = await client.post("/analyze/contract", json={
                "client_id": "test-client",
                "contract_text": sample_contract,
                "sector": "manufacturing",
            })

        assert resp.status_code == 200
        ext = resp.json()["analysis"]["extraction"]
        assert ext["client_name"] == "Unknown"
        assert ext["total_value_eur"] == 0.0
        assert ext["risk_flags"] == []

    async def test_scoring_fallback_on_failure(self, client, sample_contract, _patch_redis):
        """If scoring Predict raises, scores default to 50."""
        cd = ContractData(client_name="Test Corp")
        call_count = [0]

        def _factory(sig_cls, **kwargs):
            mock_instance = MagicMock()
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                mock_instance.return_value = _make_extraction_mock(cd)
            else:  # scoring fails
                mock_instance.side_effect = RuntimeError("Scoring LLM down")
            return mock_instance

        with patch("main.dspy.Predict", side_effect=_factory):
            resp = await client.post("/analyze/contract", json={
                "client_id": "test-client",
                "contract_text": sample_contract,
                "sector": "tech",
            })

        assert resp.status_code == 200
        sc = resp.json()["analysis"]["scoring"]
        assert sc["overall_score"] == 50
        assert sc["revenue_quality_score"] == 50

    async def test_no_risk_flags_skips_gemella(self, client, sample_contract, _patch_redis):
        """When extraction has no risk_flags, gemella array should be empty."""
        cd = ContractData(client_name="Safe Corp", risk_flags=[])

        with patch("main.dspy.Predict",
                    side_effect=_build_predict_no_gemella(cd)):
            resp = await client.post("/analyze/contract", json={
                "client_id": "c1",
                "contract_text": "Short contract.",
                "sector": "general",
            })

        assert resp.status_code == 200
        assert resp.json()["analysis"]["clausola_gemella"] == []

    async def test_long_contract_truncated(self, client, _patch_redis):
        """Contracts > 12000 chars are truncated, not rejected."""
        long_text = "A" * 20000
        cd = ContractData()
        captured_calls = []

        def _factory(sig_cls, **kwargs):
            mock_instance = MagicMock()
            def _call_fn(**kw):
                captured_calls.append(kw)
                if len(captured_calls) == 1:
                    return _make_extraction_mock(cd)
                return _make_scoring_mock()
            mock_instance.side_effect = _call_fn
            return mock_instance

        with patch("main.dspy.Predict", side_effect=_factory):
            resp = await client.post("/analyze/contract", json={
                "client_id": "c1",
                "contract_text": long_text,
                "sector": "general",
            })

        assert resp.status_code == 200
        # First call is extraction — verify truncated text
        assert len(captured_calls[0].get("contract_text", "")) <= 12000

    async def test_response_json_serializable(self, client, sample_contract, _patch_redis):
        """The entire response body must be valid JSON (no datetimes, no bytes)."""
        cd = ContractData(
            client_name="JSON Corp",
            products_services=[ProductService(name="P1", value_eur=100, recurring=True)],
            risk_flags=[RiskFlag(flag="R1", severity="low", clause="c1")],
        )

        with patch("main.dspy.Predict", side_effect=_build_predict_side_effect(cd)):
            resp = await client.post("/analyze/contract", json={
                "client_id": "c1",
                "contract_text": sample_contract,
                "sector": "tech",
            })

        body = resp.json()
        serialized = json.dumps(body)
        assert isinstance(serialized, str)

    async def test_gemella_fallback_on_failure(self, client, sample_contract, _patch_redis):
        """If gemella Predict raises, gemella list is empty — no crash."""
        cd = ContractData(
            risk_flags=[RiskFlag(flag="Test", severity="high", clause="Art 1")],
        )
        call_count = [0]

        def _factory(sig_cls, **kwargs):
            mock_instance = MagicMock()
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                mock_instance.return_value = _make_extraction_mock(cd)
            elif idx == 1:
                mock_instance.return_value = _make_scoring_mock()
            else:  # gemella fails
                mock_instance.side_effect = RuntimeError("Gemella LLM timeout")
            return mock_instance

        with patch("main.dspy.Predict", side_effect=_factory):
            resp = await client.post("/analyze/contract", json={
                "client_id": "c1",
                "contract_text": sample_contract,
                "sector": "tech",
            })

        assert resp.status_code == 200
        assert resp.json()["analysis"]["clausola_gemella"] == []
        # Scoring still works
        assert resp.json()["analysis"]["scoring"]["overall_score"] == 75

    async def test_all_three_steps_fail_gracefully(self, client, sample_contract, _patch_redis):
        """If ALL three Predicts fail, endpoint returns safe defaults everywhere."""
        def _factory(sig_cls, **kwargs):
            mock_instance = MagicMock()
            mock_instance.side_effect = RuntimeError("Total LLM failure")
            return mock_instance

        with patch("main.dspy.Predict", side_effect=_factory):
            resp = await client.post("/analyze/contract", json={
                "client_id": "c1",
                "contract_text": sample_contract,
                "sector": "general",
            })

        assert resp.status_code == 200
        body = resp.json()["analysis"]
        assert body["extraction"]["client_name"] == "Unknown"
        assert body["scoring"]["overall_score"] == 50
        assert body["clausola_gemella"] == []

    async def test_none_llm_response_handled(self, client, sample_contract, _patch_redis):
        """If LLM returns None (the original crash), no strip() error."""
        call_count = [0]

        def _factory(sig_cls, **kwargs):
            mock_instance = MagicMock()
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                # Simulate LLM returning None — the exact bug scenario
                result = MagicMock()
                result.extracted_json = None
                mock_instance.return_value = result
            else:
                mock_instance.return_value = _make_scoring_mock()
            return mock_instance

        with patch("main.dspy.Predict", side_effect=_factory):
            resp = await client.post("/analyze/contract", json={
                "client_id": "c1",
                "contract_text": sample_contract,
                "sector": "tech",
            })

        assert resp.status_code == 200
        ext = resp.json()["analysis"]["extraction"]
        assert ext["client_name"] == "Unknown"  # defaults applied

    async def test_markdown_wrapped_json_parsed(self, client, sample_contract, _patch_redis):
        """LLM wraps JSON in ```json ... ``` — parser extracts it."""
        raw_md = '```json\n{"client_name": "MD Corp", "total_value_eur": 50000}\n```'
        call_count = [0]

        def _factory(sig_cls, **kwargs):
            mock_instance = MagicMock()
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                result = MagicMock()
                result.extracted_json = raw_md
                mock_instance.return_value = result
            else:
                mock_instance.return_value = _make_scoring_mock()
            return mock_instance

        with patch("main.dspy.Predict", side_effect=_factory):
            resp = await client.post("/analyze/contract", json={
                "client_id": "c1",
                "contract_text": sample_contract,
                "sector": "tech",
            })

        assert resp.status_code == 200
        ext = resp.json()["analysis"]["extraction"]
        assert ext["client_name"] == "MD Corp"
        assert ext["total_value_eur"] == 50000.0
