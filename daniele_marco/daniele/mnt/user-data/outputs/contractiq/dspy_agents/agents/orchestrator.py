# contractiq/dspy_agents/agents/orchestrator.py
"""
ContractIQ Multi-Agent Orchestrator (MAS).

Architecture:
  1. ExtractionAgent      — Parses raw contract → structured JSON
  2. RiskAnalystAgent     — Deep risk analysis on extracted data
  3. ClausolaGemellaAgent — Benchmarks clauses vs sector corpus  [DISRUPTIVE #1]
  4. ScoringAgent         — Computes proprietary ContractIQ Score [DISRUPTIVE #2]
  5. AlertAgent           — Generates proactive push alerts       [DISRUPTIVE #3]
  6. QAAgent              — Answers ad-hoc user questions via RAG
  7. ScoreHistoryAgent    — Narrative analysis on score time-series [LOCK-IN #1]

Each client gets a COMPILED version of this orchestrator (saved to disk)
that has been optimized on their own feedback data via MIPROv2.
This compilation is the irreversible lock-in.

LLM: qwen3.5-122b via EXTERNAL_LLM_URL (OpenAI-compatible).
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import dspy
import httpx

from signatures.contract_signatures import (
    ContractExtraction,
    ClausolaGemella,
    ContractScoring,
    ProactiveAlertGeneration,
    PortfolioRiskAnalysis,
    ContractQA,
    ContractSummarization,
    ProductClassificationSignature,
    RenewalProbabilitySignature,
)

from utils.json_utils import safe_parse_json, safe_parse_int

logger = logging.getLogger(__name__)

RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag_service:8002")
_HTTPX_TIMEOUT = httpx.Timeout(30.0)


# ─────────────────────────────────────────────────────────────────────────────
# Score History Signature (defined here — not a core extraction signature)
# ─────────────────────────────────────────────────────────────────────────────

class ScoreTrendAnalysis(dspy.Signature):
    """
    Analizza la serie storica degli score ContractIQ e produce insight narrativi.
    Trasforma numeri in linguaggio che il management può citare nelle board review.
    """
    score_history: str = dspy.InputField(
        desc="JSON array di score storici con date: [{overall_score, scored_at, ...}]"
    )
    client_context: str = dspy.InputField(desc="Contesto RAG del cliente")
    trend: str = dspy.InputField(desc="improving|stable|declining calcolato numericamente")

    narrative_insight: str = dspy.OutputField(
        desc="Paragrafo narrativo sull'evoluzione del portafoglio nel tempo"
    )
    key_inflection_points: str = dspy.OutputField(
        desc="JSON array degli eventi che hanno causato variazioni significative dello score"
    )
    next_renewal_forecast: str = dspy.OutputField(
        desc="Forecast score al prossimo rinnovo principale, con motivazione"
    )
    board_summary: str = dspy.OutputField(
        desc="Esattamente 2 frasi adatte a una board review executive"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Individual Agents
# ─────────────────────────────────────────────────────────────────────────────

class ContractSummarizer(dspy.Module):
    """Summarizes contract to fit LLM effective context length."""
    def __init__(self) -> None:
        super().__init__()
        self.summarize = dspy.Predict(ContractSummarization)

    def forward(self, contract_text: str) -> dspy.Prediction:
        return self.summarize(contract_text=contract_text)


class ExtractionAgent(dspy.Module):
    """Agent 1: Extract structured data from raw contract text."""

    def __init__(self) -> None:
        super().__init__()
        self.extract = dspy.ChainOfThought(ContractExtraction)

    def forward(self, contract_summary: str, client_context: str) -> dspy.Prediction:
        return self.extract(
            contract_summary=contract_summary,
            client_context=client_context,
        )


class ProductClassificationAgent(dspy.Module):
    """Agent 1b: Normalize free-form product/service strings into standard categories."""

    def __init__(self) -> None:
        super().__init__()
        self.classify = dspy.ChainOfThought(ProductClassificationSignature)

    def forward(self, raw_product_string: str, client_context: str) -> dspy.Prediction:
        return self.classify(
            raw_product_string=raw_product_string,
            client_context=client_context,
        )


class RiskAnalystAgent(dspy.Module):
    """Agent 2: Perform portfolio-level risk analysis."""

    def __init__(self) -> None:
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
    Respects is_benchmark_available flag — never produces fake percentiles.
    """

    def __init__(self) -> None:
        super().__init__()
        self.benchmark = dspy.ChainOfThought(ClausolaGemella)

    def forward(
        self,
        clause_text: str,
        clause_type: str,
        sector: str,
        benchmark_corpus: str,
        is_benchmark_available: bool = True,
    ) -> dspy.Prediction:
        return self.benchmark(
            clause_text=clause_text,
            clause_type=clause_type,
            sector=sector,
            benchmark_corpus=benchmark_corpus,
            is_benchmark_available=is_benchmark_available,
        )


class ScoringAgent(dspy.Module):
    """
    Agent 4 [DISRUPTIVE MOVE #2]: Compute proprietary ContractIQ Score.
    Single number boards can act on. Defensible, brandable, citable.
    Accepts score_history to compute delta vs previous scoring.
    """

    def __init__(self) -> None:
        super().__init__()
        self.score = dspy.ChainOfThought(ContractScoring)

    def forward(
        self,
        extracted_contract: str,
        portfolio_context: str,
        sector_benchmarks: str,
        score_history: str = "",
    ) -> dspy.Prediction:
        return self.score(
            extracted_contract=extracted_contract,
            portfolio_context=portfolio_context,
            sector_benchmarks=sector_benchmarks,
            score_history=score_history,
        )


class RenewalProbabilityAgent(dspy.Module):
    """
    Agent 4b: Calculate probability of renewal (0-100)
    Read client history and returns a probabilistic assessment.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prob = dspy.ChainOfThought(RenewalProbabilitySignature)

    def forward(
        self,
        contract_data: str,
        client_history: str,
        portfolio_context: str,
    ) -> dspy.Prediction:
        return self.prob(
            contract_data=contract_data,
            client_history=client_history,
            portfolio_context=portfolio_context,
        )


class AlertAgent(dspy.Module):
    """
    Agent 5 [DISRUPTIVE MOVE #3]: Generate proactive push alerts.
    Transforms ContractIQ from BI tool → Chief Contract Officer.
    """

    def __init__(self) -> None:
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
    """
    Agent 6: Answer ad-hoc questions with full citation tracking.

    BUG FIX: replaced dspy.ReAct(ContractQA, tools=[]) with
    dspy.ChainOfThought(ContractQA). ReAct with empty tools list at
    __init__ time prevents MIPROv2 from optimizing tool calls correctly.
    Context is already retrieved upstream and passed via retrieved_context.
    """

    def __init__(self) -> None:
        super().__init__()
        self.qa = dspy.ChainOfThought(ContractQA)

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


class ScoreHistoryAgent(dspy.Module):
    """
    Agent 7 [LOCK-IN #1]: Narrative analysis on ContractIQ Score time-series.

    Transforms raw score history rows into board-ready language.
    The longer the history, the richer the insight — irreproducible by competitors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.analyze = dspy.ChainOfThought(ScoreTrendAnalysis)

    def forward(
        self,
        score_history: str,
        client_context: str,
        trend: str,
    ) -> dspy.Prediction:
        return self.analyze(
            score_history=score_history,
            client_context=client_context,
            trend=trend,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MASTER ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class ContractIQOrchestrator(dspy.Module):
    """
    Master Multi-Agent Orchestrator for ContractIQ.

    Compiled per-client via DSPy MIPROv2 optimizer using user feedback
    as training signal. The compiled .json saved to disk encodes
    client-specific prompt instructions — the product moat.

    LLM backend: qwen3.5-122b via OpenAI-compatible EXTERNAL_LLM_URL.
    No Anthropic, no Voyage AI, no Ollama dependencies.
    """

    def __init__(self) -> None:
        super().__init__()
        self.summarizer = ContractSummarizer()
        self.extractor = ExtractionAgent()
        self.product_classifier = ProductClassificationAgent()
        self.risk_analyst = RiskAnalystAgent()
        self.clausola_gemella = ClausolaGemellaAgent()
        self.scorer = ScoringAgent()
        self.renewal_agent = RenewalProbabilityAgent()
        self.alert_agent = AlertAgent()
        self.qa_agent = QAAgent()
        self.score_history_agent = ScoreHistoryAgent()

    # ── Contract Analysis ─────────────────────────────────────────────────────

    def analyze_contract(
        self,
        contract_text: str,
        client_context: str,
        portfolio_context: str,
        sector_benchmarks: str,
        benchmark_corpus: str,
        sector: str,
        current_date: str,
        client_id: str = "",
        contract_id: str = "",
        score_history: str = "",
        is_benchmark_available: bool = True,
    ) -> dict[str, Any]:
        """
        Full analysis pipeline for a single contract.
        Runs agents in dependency order.
        Persists score to RAG service history endpoint (fire-and-forget).
        """
        logger.info("Starting contract analysis pipeline client=%s contract=%s", client_id, contract_id)

        # Step 0: Summarize contract text
        summary_pred = self.summarizer(contract_text=contract_text)
        summarized_text = summary_pred.summary
        logger.info("Summarization complete client=%s", client_id)

        # Step 1: Extract structured data
        extracted = self.extractor(
            contract_summary=summarized_text,
            client_context=client_context,
        )
        logger.info("Extraction complete client=%s", client_id)

        # Step 1b: Normalize products/services
        products = safe_parse_json(extracted.products_services)
        if isinstance(products, list):
            for prod in products:
                raw_name = prod.get("name", "")
                if raw_name:
                    p_res = self.product_classifier(
                        raw_product_string=raw_name,
                        client_context=client_context
                    )
                    prod["name"] = p_res.standard_category
        
        # Step 2: Score the contract (with historical context)
        contract_data_for_scoring = json.dumps({
            "client_name": extracted.client_name,
            "total_value_eur": extracted.total_value_eur,
            "arr": extracted.annual_recurring_revenue,
            "end_date": extracted.end_date,
            "discount_pct": extracted.discount_percentage,
            "payment_terms": extracted.payment_terms_days,
            "risk_flags": extracted.risk_flags,
            "auto_renewal": extracted.auto_renewal,
        })
        
        scored = self.scorer(
            extracted_contract=contract_data_for_scoring,
            portfolio_context=portfolio_context,
            sector_benchmarks=sector_benchmarks,
            score_history=score_history,
        )
        logger.info("Score computed: %d (delta: %d) client=%s", scored.overall_score, scored.score_delta, client_id)

        # Step 2c: Renewal Probability
        renewal_prob = self.renewal_agent(
            contract_data=contract_data_for_scoring,
            client_history=client_context,
            portfolio_context=portfolio_context,
        )
        logger.info("Renewal Probability computed: %s client=%s", renewal_prob.renewal_probability, client_id)

        # Step 2b: Persist score to history (fire-and-forget — never blocks the pipeline)
        if client_id and contract_id:
            self._persist_score_async(client_id, contract_id, scored)

        # Step 3: Benchmark key clauses via Clausola Gemella
        clause_benchmarks = []
        try:
            risk_flags = safe_parse_json(extracted.risk_flags)
            if isinstance(risk_flags, list):
                for flag in risk_flags[:3]:
                    benchmark = self.clausola_gemella(
                        clause_text=flag.get("clause", ""),
                        clause_type=flag.get("flag", "general"),
                        sector=sector,
                        benchmark_corpus=benchmark_corpus,
                        is_benchmark_available=is_benchmark_available,
                    )
                    clause_benchmarks.append({
                        "clause_type": flag.get("flag"),
                        "percentile": benchmark.percentile_ranking,
                        "summary": benchmark.benchmark_summary,
                        "negotiation_script": benchmark.negotiation_script,
                        "priority": benchmark.priority,
                        "benchmark_reliability": benchmark.benchmark_reliability,
                    })
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("Clausola Gemella partial failure: %s", exc)

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
                "renewal_probability": safe_parse_int(renewal_prob.renewal_probability),
                "products_services": products if isinstance(products, list) else safe_parse_json(extracted.products_services),
                "penalty_clauses": safe_parse_json(extracted.penalty_clauses),
                "risk_flags": safe_parse_json(extracted.risk_flags),
                "non_standard_terms": safe_parse_json(extracted.non_standard_terms),
            },
            "scoring": {
                "overall_score": safe_parse_int(scored.overall_score),
                "revenue_quality": safe_parse_int(scored.revenue_quality_score),
                "risk_exposure": safe_parse_int(scored.risk_exposure_score),
                "strategic_value": safe_parse_int(scored.strategic_value_score),
                "concentration_risk": safe_parse_int(scored.concentration_risk_score),
                "score_delta": safe_parse_int(scored.score_delta, 0),
                "rationale": scored.score_rationale,
                "improvement_actions": safe_parse_json(scored.improvement_actions),
            },
            "clausola_gemella": clause_benchmarks,
        }

    def _persist_score_async(
        self, client_id: str, contract_id: str, scored: dspy.Prediction
    ) -> None:
        """
        Fire-and-forget score persistence to RAG service.
        Uses a synchronous httpx call wrapped in try/except so it
        never blocks or raises in the main analysis pipeline.
        """
        payload = {
            "client_id": client_id,
            "contract_id": contract_id,
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "score_data": {
                "overall_score": scored.overall_score,
                "revenue_quality": getattr(scored, "revenue_quality_score", None),
                "risk_exposure": getattr(scored, "risk_exposure_score", None),
                "strategic_value": getattr(scored, "strategic_value_score", None),
                "concentration_risk": getattr(scored, "concentration_risk_score", None),
                "score_delta": getattr(scored, "score_delta", 0),
                "rationale": getattr(scored, "score_rationale", ""),
            },
        }
        try:
            with httpx.Client(timeout=_HTTPX_TIMEOUT) as client:
                resp = client.post(f"{RAG_SERVICE_URL}/score/history/store", json=payload)
                if resp.status_code != 200:
                    logger.warning(
                        "Score history store returned %d for client=%s contract=%s",
                        resp.status_code, client_id, contract_id,
                    )
        except Exception as exc:
            logger.warning(
                "Score history persistence failed (non-blocking) client=%s: %s",
                client_id, exc,
            )

    # ── Portfolio Analysis ────────────────────────────────────────────────────

    def analyze_portfolio(
        self,
        portfolio_data: str,
        client_history: str,
        portfolio_context: str,
        current_date: str,
    ) -> dict[str, Any]:
        """Portfolio-level analysis: risks + proactive alerts."""

        risk = self.risk_analyst(
            portfolio_data=portfolio_data,
            client_history=client_history,
        )

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
                "critical": safe_parse_json(alerts.critical_alerts),
                "opportunities": safe_parse_json(alerts.opportunity_alerts),
                "weekly_priorities": alerts.weekly_priorities,
            },
        }

    # ── Score History Analysis ────────────────────────────────────────────────

    def analyze_score_history(
        self,
        score_history: str,
        client_context: str,
        trend: str,
    ) -> dict[str, Any]:
        """
        Generate narrative insights from the ContractIQ Score time-series.
        Produces board-ready language from raw score history data.
        The longer the history, the deeper the insight — the stronger the lock-in.
        """
        result = self.score_history_agent(
            score_history=score_history,
            client_context=client_context,
            trend=trend,
        )
        return {
            "narrative_insight": result.narrative_insight,
            "key_inflection_points": result.key_inflection_points,
            "next_renewal_forecast": result.next_renewal_forecast,
            "board_summary": result.board_summary,
            "trend": trend,
        }

    # ── QA ────────────────────────────────────────────────────────────────────

    def answer_question(
        self,
        question: str,
        retrieved_context: str,
        client_history: str,
        user_feedback_context: str = "",
    ) -> dict[str, Any]:
        """
        Answer an ad-hoc question about the contract portfolio.

        user_feedback_context: injected by the gateway from
        /user/feedback/context/{client_id} — contains relevant past
        interactions as prompt context. The more a user interacts,
        the more personalized future answers become (interaction lock-in).
        """
        enriched_history = (
            (user_feedback_context + "\n\n" + client_history).strip()
            if user_feedback_context
            else client_history
        )

        result = self.qa_agent(
            question=question,
            retrieved_context=retrieved_context,
            client_history=enriched_history,
        )
        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": safe_parse_json(result.sources),
            "follow_up_questions": safe_parse_json(result.follow_up_questions),
        }