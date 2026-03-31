"""
contractiq/dspy_agents/agents/orchestrator.py

ContractIQ Multi-Agent Orchestrator (MAS).

Architecture:
  1. ExtractionAgent    — Parses raw contract → structured JSON
  2. RiskAnalystAgent   — Deep risk analysis on extracted data
  3. ClausolaGemella    — Benchmarks clauses vs sector corpus  [DISRUPTIVE #1]
  4. ScoringAgent       — Computes proprietary ContractIQ Score [DISRUPTIVE #2]
  5. AlertAgent         — Generates proactive push alerts       [DISRUPTIVE #3]
  6. QAAgent            — Answers ad-hoc user questions via RAG

Each client gets a COMPILED version of this orchestrator (saved to disk)
that has been optimized on their own feedback data via MIPROv2.
This compilation is the irreversible lock-in.
"""

import json
import logging
from typing import Any

import dspy

from signatures.contract_signatures import (
    ContractExtraction,
    ClausolaGemella,
    ContractScoring,
    ProactiveAlertGeneration,
    PortfolioRiskAnalysis,
    ContractQA,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Individual Agents
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionAgent(dspy.Module):
    """Agent 1: Extract structured data from raw contract text."""

    def __init__(self):
        super().__init__()
        # ChainOfThought adds step-by-step reasoning before output
        self.extract = dspy.ChainOfThought(ContractExtraction)

    def forward(self, contract_text: str, client_context: str) -> dspy.Prediction:
        return self.extract(
            contract_text=contract_text,
            client_context=client_context,
        )


class RiskAnalystAgent(dspy.Module):
    """Agent 2: Perform portfolio-level risk analysis."""

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(PortfolioRiskAnalysis)

    def forward(self, portfolio_data: str, client_history: str) -> dspy.Prediction:
        return self.analyze(
            portfolio_data=portfolio_data,
            client_history=client_history,
        )


class ClausolaGemellaAgent(dspy.Module):
    """
    Agent 3 [DISRUPTIVE MOVE #1]: Benchmark clauses against sector corpus.
    'Questo SLA è nel 12% peggiore del tuo settore.'
    """

    def __init__(self):
        super().__init__()
        self.benchmark = dspy.ChainOfThought(ClausolaGemella)

    def forward(
        self,
        clause_text: str,
        clause_type: str,
        sector: str,
        benchmark_corpus: str,
    ) -> dspy.Prediction:
        return self.benchmark(
            clause_text=clause_text,
            clause_type=clause_type,
            sector=sector,
            benchmark_corpus=benchmark_corpus,
        )


class ScoringAgent(dspy.Module):
    """
    Agent 4 [DISRUPTIVE MOVE #2]: Compute proprietary ContractIQ Score.
    Single number boards can act on. Defensible, brandable, citable.
    """

    def __init__(self):
        super().__init__()
        self.score = dspy.ChainOfThought(ContractScoring)

    def forward(
        self,
        extracted_contract: str,
        portfolio_context: str,
        sector_benchmarks: str,
    ) -> dspy.Prediction:
        return self.score(
            extracted_contract=extracted_contract,
            portfolio_context=portfolio_context,
            sector_benchmarks=sector_benchmarks,
        )


class AlertAgent(dspy.Module):
    """
    Agent 5 [DISRUPTIVE MOVE #3]: Generate proactive push alerts.
    Transforms ContractIQ from BI tool → Chief Contract Officer.
    """

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ProactiveAlertGeneration)

    def forward(
        self,
        portfolio_snapshot: str,
        historical_patterns: str,
        current_date: str,
    ) -> dspy.Prediction:
        return self.generate(
            portfolio_snapshot=portfolio_snapshot,
            historical_patterns=historical_patterns,
            current_date=current_date,
        )


class QAAgent(dspy.Module):
    """Agent 6: Answer ad-hoc questions with full citation tracking."""

    def __init__(self):
        super().__init__()
        # ReAct: Reason + Act (can call retrieval iteratively)
        self.qa = dspy.ReAct(
            ContractQA,
            tools=[],  # Tools injected at runtime via RAG service
        )

    def forward(
        self,
        question: str,
        retrieved_context: str,
        client_history: str,
    ) -> dspy.Prediction:
        return self.qa(
            question=question,
            retrieved_context=retrieved_context,
            client_history=client_history,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MASTER ORCHESTRATOR
# Coordinates all agents in the right sequence
# ─────────────────────────────────────────────────────────────────────────────

class ContractIQOrchestrator(dspy.Module):
    """
    Master Multi-Agent Orchestrator for ContractIQ.

    This is compiled per-client via DSPy MIPROv2 optimizer
    using user feedback as training signal.

    The compiled .json file saved to disk is the product moat:
    it encodes client-specific prompt instructions that took
    real usage data to produce — impossible to replicate externally.
    """

    def __init__(self):
        super().__init__()
        self.extractor = ExtractionAgent()
        self.risk_analyst = RiskAnalystAgent()
        self.clausola_gemella = ClausolaGemellaAgent()
        self.scorer = ScoringAgent()
        self.alert_agent = AlertAgent()
        self.qa_agent = QAAgent()

    def analyze_contract(
        self,
        contract_text: str,
        client_context: str,
        portfolio_context: str,
        sector_benchmarks: str,
        benchmark_corpus: str,
        sector: str,
        current_date: str,
    ) -> dict[str, Any]:
        """
        Full analysis pipeline for a single contract.
        Runs agents in dependency order.
        """
        logger.info("Starting contract analysis pipeline")

        # Step 1: Extract structured data
        extracted = self.extractor(
            contract_text=contract_text,
            client_context=client_context,
        )
        logger.info("Extraction complete")

        # Step 2: Score the contract
        scored = self.scorer(
            extracted_contract=json.dumps({
                "client_name": extracted.client_name,
                "total_value_eur": extracted.total_value_eur,
                "arr": extracted.annual_recurring_revenue,
                "end_date": extracted.end_date,
                "discount_pct": extracted.discount_percentage,
                "payment_terms": extracted.payment_terms_days,
                "risk_flags": extracted.risk_flags,
                "auto_renewal": extracted.auto_renewal,
            }),
            portfolio_context=portfolio_context,
            sector_benchmarks=sector_benchmarks,
        )
        logger.info(f"Score computed: {scored.overall_score}")

        # Step 3: Benchmark key clauses via Clausola Gemella
        clause_benchmarks = []
        try:
            risk_flags = json.loads(extracted.risk_flags or "[]")
            for flag in risk_flags[:3]:  # Top 3 risky clauses
                benchmark = self.clausola_gemella(
                    clause_text=flag.get("clause", ""),
                    clause_type=flag.get("flag", "general"),
                    sector=sector,
                    benchmark_corpus=benchmark_corpus,
                )
                clause_benchmarks.append({
                    "clause_type": flag.get("flag"),
                    "percentile": benchmark.percentile_ranking,
                    "summary": benchmark.benchmark_summary,
                    "negotiation_script": benchmark.negotiation_script,
                    "priority": benchmark.priority,
                })
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Clausola Gemella partial failure: {e}")

        return {
            "extraction": {
                "client_name": extracted.client_name,
                "contract_type": extracted.contract_type,
                "total_value_eur": extracted.total_value_eur,
                "arr": extracted.annual_recurring_revenue,
                "start_date": extracted.start_date,
                "end_date": extracted.end_date,
                "auto_renewal": extracted.auto_renewal,
                "renewal_notice_days": extracted.renewal_notice_days,
                "payment_terms_days": extracted.payment_terms_days,
                "discount_percentage": extracted.discount_percentage,
                "sla_uptime_percentage": extracted.sla_uptime_percentage,
                "products_services": extracted.products_services,
                "penalty_clauses": extracted.penalty_clauses,
                "risk_flags": extracted.risk_flags,
                "non_standard_terms": extracted.non_standard_terms,
            },
            "scoring": {
                "overall_score": scored.overall_score,
                "revenue_quality": scored.revenue_quality_score,
                "risk_exposure": scored.risk_exposure_score,
                "strategic_value": scored.strategic_value_score,
                "concentration_risk": scored.concentration_risk_score,
                "rationale": scored.score_rationale,
                "improvement_actions": scored.improvement_actions,
            },
            "clausola_gemella": clause_benchmarks,
        }

    def analyze_portfolio(
        self,
        portfolio_data: str,
        client_history: str,
        portfolio_context: str,
        current_date: str,
    ) -> dict[str, Any]:
        """Portfolio-level analysis: risks + alerts."""

        # Risk analysis
        risk = self.risk_analyst(
            portfolio_data=portfolio_data,
            client_history=client_history,
        )

        # Proactive alerts
        alerts = self.alert_agent(
            portfolio_snapshot=portfolio_data,
            historical_patterns=client_history,
            current_date=current_date,
        )

        return {
            "risk_analysis": {
                "concentration_risk": risk.concentration_risk,
                "expiry_risk": risk.expiry_risk,
                "clause_risk_summary": risk.clause_risk_summary,
                "systemic_risks": risk.systemic_risks,
                "mitigation_plan": risk.mitigation_plan,
            },
            "proactive_alerts": {
                "critical": alerts.critical_alerts,
                "opportunities": alerts.opportunity_alerts,
                "weekly_priorities": alerts.weekly_priorities,
            },
        }

    def answer_question(
        self,
        question: str,
        retrieved_context: str,
        client_history: str,
    ) -> dict[str, Any]:
        """Answer an ad-hoc question about the contract portfolio."""
        result = self.qa_agent(
            question=question,
            retrieved_context=retrieved_context,
            client_history=client_history,
        )
        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": result.sources,
            "follow_up_questions": result.follow_up_questions,
        }
