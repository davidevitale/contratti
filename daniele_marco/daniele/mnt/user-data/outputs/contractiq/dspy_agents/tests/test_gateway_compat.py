"""
Test 7 — Gateway compatibility.
Ensures the /analyze/contract response has the exact shape
that gateway/main.py expects at lines 254-276.
All three steps now use dspy.Predict + manual JSON parsing.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from main import ContractData, ProductService, RiskFlag, ScoringData, GemellaData


def _make_extraction_mock(cd):
    pred = MagicMock()
    pred.extracted_json = json.dumps(cd.model_dump())
    return pred


def _make_scoring_mock(sd=None):
    pred = MagicMock()
    pred.scoring_json = json.dumps((sd or ScoringData()).model_dump())
    return pred


def _make_gemella_mock(gd=None):
    pred = MagicMock()
    pred.gemella_json = json.dumps((gd or GemellaData()).model_dump())
    return pred


def _build_side_effect(cd, scoring=None, gemella=None):
    """Build a Predict side_effect for extraction → scoring → gemella."""
    call_count = [0]

    def _factory(sig_cls, **kwargs):
        mock = MagicMock()
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:  # extraction
            mock.return_value = _make_extraction_mock(cd)
        elif idx == 1:  # scoring
            mock.return_value = _make_scoring_mock(scoring)
        else:  # gemella
            mock.return_value = _make_gemella_mock(gemella)
        return mock

    return _factory


@pytest.mark.asyncio
class TestGatewayCompatibility:

    async def test_extraction_keys_match_gateway_expectations(self, client, sample_contract, _patch_redis):
        """Gateway reads: analysis.extraction.{client_name, contract_type, total_value_eur, ...}"""
        cd = ContractData(
            client_name="GW Corp",
            contract_type="SaaS",
            total_value_eur=100000,
            end_date="2027-12-31",
        )
        sd = ScoringData(overall_score=70, revenue_quality_score=80)

        with patch("main.dspy.Predict", side_effect=_build_side_effect(cd, sd)):
            resp = await client.post("/analyze/contract", json={
                "client_id": "gw",
                "contract_text": sample_contract,
                "sector": "tech",
            })

        body = resp.json()
        ext = body["analysis"]["extraction"]

        gateway_required_keys = [
            "client_name", "contract_type", "total_value_eur",
            "end_date", "risk_flags",
        ]
        for key in gateway_required_keys:
            assert key in ext, f"Missing key '{key}' required by gateway"

        sc = body["analysis"]["scoring"]
        assert "overall_score" in sc
        assert isinstance(sc["overall_score"], int)

    async def test_risk_flags_is_list_of_dicts(self, client, sample_contract, _patch_redis):
        """risk_flags is a native list of dicts. JSON roundtrip must survive."""
        cd = ContractData(
            risk_flags=[RiskFlag(flag="Exclusive", severity="high", clause="Art 6.2")],
        )
        gd = GemellaData(percentile_ranking=50, priority="low")

        with patch("main.dspy.Predict", side_effect=_build_side_effect(cd, gemella=gd)):
            resp = await client.post("/analyze/contract", json={
                "client_id": "gw",
                "contract_text": sample_contract,
                "sector": "tech",
            })

        flags = resp.json()["analysis"]["extraction"]["risk_flags"]
        assert isinstance(flags, list)
        assert isinstance(flags[0], dict)
        assert flags[0]["flag"] == "Exclusive"

        roundtrip = json.loads(json.dumps(flags))
        assert roundtrip == flags

    async def test_clausola_gemella_structure(self, client, sample_contract, _patch_redis):
        """Gateway reads analysis.clausola_gemella as a list of objects."""
        cd = ContractData(
            risk_flags=[RiskFlag(flag="R", severity="medium", clause="c")],
        )
        gd = GemellaData(
            percentile_ranking=75,
            benchmark_summary="Above avg",
            deviation_analysis="Higher cap",
            negotiation_script="Negotiate",
            priority="high",
        )

        with patch("main.dspy.Predict", side_effect=_build_side_effect(cd, gemella=gd)):
            resp = await client.post("/analyze/contract", json={
                "client_id": "gw",
                "contract_text": sample_contract,
                "sector": "manufacturing",
            })

        gem = resp.json()["analysis"]["clausola_gemella"]
        assert isinstance(gem, list)
        assert len(gem) == 1
        assert set(gem[0].keys()) == {
            "percentile_ranking", "benchmark_summary",
            "deviation_analysis", "negotiation_script", "priority",
        }
