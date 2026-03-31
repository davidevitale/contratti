"""
Test 1 — Pydantic models: defaults, validation, serialization.
Verifies that ContractData never produces None fields and
survives malformed / partial input gracefully.
"""

import json
import pytest
from main import (
    ContractData, ProductService, PenaltyClause, RiskFlag,
    ScoringData,
    GemellaData,
)


# ────────────────────────────────────────────────────────────────────────────
# ContractData defaults
# ────────────────────────────────────────────────────────────────────────────

class TestContractDataDefaults:
    """Every field must have a safe, non-None default."""

    def test_empty_construction(self):
        cd = ContractData()
        assert cd.client_name == "Unknown"
        assert cd.contract_type == "Agreement"
        assert cd.total_value_eur == 0.0
        assert cd.annual_recurring_revenue == 0.0
        assert cd.start_date == ""
        assert cd.end_date == ""
        assert cd.auto_renewal is False
        assert cd.renewal_notice_days == 30
        assert cd.payment_terms_days == 30
        assert cd.discount_percentage == 0.0
        assert cd.sla_uptime_percentage == 0.0
        assert cd.products_services == []
        assert cd.penalty_clauses == []
        assert cd.risk_flags == []
        assert cd.non_standard_terms == []

    def test_no_none_in_model_dump(self):
        """model_dump() must never contain None — gateway relies on this."""
        dump = ContractData().model_dump()
        for key, value in dump.items():
            assert value is not None, f"{key} is None in model_dump()"

    def test_model_dump_is_json_serializable(self):
        """The dict returned by model_dump() must survive json.dumps."""
        dump = ContractData().model_dump()
        serialized = json.dumps(dump)
        assert isinstance(serialized, str)
        roundtrip = json.loads(serialized)
        assert roundtrip["client_name"] == "Unknown"


# ────────────────────────────────────────────────────────────────────────────
# ContractData with realistic data
# ────────────────────────────────────────────────────────────────────────────

class TestContractDataPopulated:
    def test_full_population(self):
        cd = ContractData(
            client_name="Acme S.p.A.",
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
        dump = cd.model_dump()
        assert dump["client_name"] == "Acme S.p.A."
        assert dump["total_value_eur"] == 210000.0
        assert len(dump["products_services"]) == 1
        assert dump["products_services"][0]["name"] == "FactoryOS Core"
        assert dump["risk_flags"][0]["severity"] == "high"

    def test_partial_population_fills_defaults(self):
        cd = ContractData(client_name="Test Corp")
        assert cd.client_name == "Test Corp"
        assert cd.total_value_eur == 0.0  # default
        assert cd.risk_flags == []        # default


# ────────────────────────────────────────────────────────────────────────────
# Sub-models
# ────────────────────────────────────────────────────────────────────────────

class TestSubModels:
    def test_product_service_defaults(self):
        ps = ProductService()
        assert ps.name == ""
        assert ps.value_eur == 0.0
        assert ps.recurring is False

    def test_penalty_clause_defaults(self):
        pc = PenaltyClause()
        assert pc.trigger == ""
        assert pc.amount_eur == 0.0

    def test_risk_flag_defaults(self):
        rf = RiskFlag()
        assert rf.flag == ""
        assert rf.severity == "medium"
        assert rf.clause == ""

    def test_risk_flag_as_dict(self):
        rf = RiskFlag(flag="Exclusivity", severity="high", clause="Art 6")
        d = rf.model_dump()
        assert d == {"flag": "Exclusivity", "severity": "high", "clause": "Art 6"}


# ────────────────────────────────────────────────────────────────────────────
# Edge cases — robust parsing
# ────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_coerce_int_to_float(self):
        """Pydantic should accept int where float is expected."""
        cd = ContractData(total_value_eur=100000)
        assert cd.total_value_eur == 100000.0

    def test_empty_lists_remain_lists(self):
        cd = ContractData(products_services=[], risk_flags=[])
        dump = cd.model_dump()
        assert isinstance(dump["products_services"], list)
        assert isinstance(dump["risk_flags"], list)

    def test_nested_model_roundtrip(self):
        """Serialize → deserialize must preserve nested structure."""
        cd = ContractData(
            products_services=[ProductService(name="X", value_eur=1000, recurring=True)],
            risk_flags=[RiskFlag(flag="Y", severity="low", clause="Z")],
        )
        raw = json.dumps(cd.model_dump())
        rebuilt = ContractData(**json.loads(raw))
        assert rebuilt.products_services[0].name == "X"
        assert rebuilt.risk_flags[0].flag == "Y"


# ────────────────────────────────────────────────────────────────────────────
# ScoringData defaults & serialization
# ────────────────────────────────────────────────────────────────────────────

class TestScoringDataDefaults:
    """All ScoringData fields must have safe defaults."""

    def test_empty_construction(self):
        sd = ScoringData()
        assert sd.overall_score == 50
        assert sd.revenue_quality_score == 50
        assert sd.risk_exposure_score == 50
        assert sd.strategic_value_score == 50
        assert sd.concentration_risk_score == 50
        assert sd.score_rationale == ""
        assert sd.top_score_drivers == []
        assert sd.improvement_actions == []

    def test_no_none_in_model_dump(self):
        dump = ScoringData().model_dump()
        for key, value in dump.items():
            assert value is not None, f"ScoringData.{key} is None"

    def test_populated_scoring(self):
        sd = ScoringData(
            overall_score=85,
            revenue_quality_score=90,
            score_rationale="Strong recurring revenue",
            top_score_drivers=["ARR growth", "Long-term contract"],
            improvement_actions=["Reduce payment terms", "Add SLA penalties"],
        )
        dump = sd.model_dump()
        assert dump["overall_score"] == 85
        assert len(dump["top_score_drivers"]) == 2
        assert dump["top_score_drivers"][0] == "ARR growth"
        assert dump["improvement_actions"][1] == "Add SLA penalties"

    def test_string_lists_accept_plain_strings(self):
        """Qwen sends plain strings — List[str] accepts them without error."""
        sd = ScoringData(
            top_score_drivers=["high ARR", "low risk"],
            improvement_actions=["renegotiate"],
        )
        assert isinstance(sd.top_score_drivers, list)
        assert all(isinstance(s, str) for s in sd.top_score_drivers)


# ────────────────────────────────────────────────────────────────────────────
# GemellaData defaults & serialization
# ────────────────────────────────────────────────────────────────────────────

class TestGemellaDataDefaults:
    """All GemellaData fields must have safe defaults."""

    def test_empty_construction(self):
        gd = GemellaData()
        assert gd.percentile_ranking == 50
        assert gd.benchmark_summary == ""
        assert gd.deviation_analysis == ""
        assert gd.negotiation_script == ""
        assert gd.priority == "medium"

    def test_no_none_in_model_dump(self):
        dump = GemellaData().model_dump()
        for key, value in dump.items():
            assert value is not None, f"GemellaData.{key} is None"

    def test_populated_gemella(self):
        gd = GemellaData(
            percentile_ranking=25,
            benchmark_summary="Below sector avg",
            deviation_analysis="Higher liability cap",
            negotiation_script="Push for standard cap",
            priority="urgent",
        )
        dump = gd.model_dump()
        assert dump["percentile_ranking"] == 25
        assert dump["priority"] == "urgent"

    def test_json_roundtrip(self):
        gd = GemellaData(percentile_ranking=75, priority="high")
        raw = json.dumps(gd.model_dump())
        rebuilt = GemellaData(**json.loads(raw))
        assert rebuilt.percentile_ranking == 75
        assert rebuilt.priority == "high"


# ────────────────────────────────────────────────────────────────────────────
# extract_json_from_response — robust JSON parser
# ────────────────────────────────────────────────────────────────────────────

from main import extract_json_from_response


class TestExtractJsonFromResponse:
    """Verify the parser handles all LLM response formats."""

    def test_none_returns_empty(self):
        assert extract_json_from_response(None) == {}

    def test_empty_string_returns_empty(self):
        assert extract_json_from_response("") == {}

    def test_pure_json(self):
        result = extract_json_from_response('{"client_name": "Acme"}')
        assert result["client_name"] == "Acme"

    def test_markdown_code_block(self):
        raw = '```json\n{"client_name": "MD Corp"}\n```'
        result = extract_json_from_response(raw)
        assert result["client_name"] == "MD Corp"

    def test_text_before_json(self):
        raw = 'Here is the result:\n{"client_name": "Extra"}'
        result = extract_json_from_response(raw)
        assert result["client_name"] == "Extra"

    def test_text_after_json(self):
        raw = '{"client_name": "After"}\nHope this helps!'
        result = extract_json_from_response(raw)
        assert result["client_name"] == "After"

    def test_non_string_returns_empty(self):
        assert extract_json_from_response(12345) == {}
        assert extract_json_from_response([1, 2]) == {}

    def test_garbage_text_returns_empty(self):
        assert extract_json_from_response("I don't know how to help") == {}
