# contractiq/dspy_agents/signatures/contract_signatures.py
"""
All DSPy Signatures for ContractIQ agents.
Signatures are the structured I/O contracts that DSPy optimizes.
The optimizer learns the best prompts FOR EACH CLIENT — this is the lock-in.
"""

import dspy


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION SIGNATURES
# ─────────────────────────────────────────────────────────────────────────────

class ContractSummarization(dspy.Signature):
    """
    Produce a summary of MAX 600 words focused on: parties involved, key obligations, payment terms, duration/deadlines, penalties, termination clauses, governing law.
    """
    contract_text: str = dspy.InputField(desc="Full text of the contract to summarize")
    summary: str = dspy.OutputField(desc="Summary of the contract (max 600 words)")

class ContractExtraction(dspy.Signature):
    """
    Extract all structured fields from a contract document.
    Use the client context to identify patterns specific to this client's
    historical contracts and flag deviations from their standard terms.
    """
    contract_summary: str = dspy.InputField(desc="Summary of the contract to analyze")
    client_context: str = dspy.InputField(
        desc="Client-specific RAG context: past contracts, preferences, standard terms"
    )

    client_name: str = dspy.OutputField(desc="Name of the counterpart client/company")
    contract_type: str = dspy.OutputField(desc="Type: SaaS, Services, Supply, License, etc.")
    total_value_eur: float = dspy.OutputField(desc="Total contract value in EUR, 0 if not specified")
    annual_recurring_revenue: float = dspy.OutputField(desc="ARR in EUR if recurring, else 0")
    start_date: str = dspy.OutputField(desc="Contract start date ISO format YYYY-MM-DD")
    end_date: str = dspy.OutputField(desc="Contract end date ISO format YYYY-MM-DD")
    auto_renewal: bool = dspy.OutputField(desc="True if contract has automatic renewal clause")
    renewal_notice_days: int = dspy.OutputField(desc="Days notice required to block renewal, 0 if none")
    payment_terms_days: int = dspy.OutputField(desc="Payment terms in days (e.g., 30, 60, 90)")
    discount_percentage: float = dspy.OutputField(desc="Applied discount percentage, 0 if none")
    sla_uptime_percentage: float = dspy.OutputField(desc="SLA uptime commitment %, 0 if none")

    products_services: str = dspy.OutputField(
        desc='JSON array of products/services: [{"name": str, "value_eur": float, "recurring": bool}]'
    )
    penalty_clauses: str = dspy.OutputField(
        desc='Lista delle penali. TASSATIVO: DEVE ESSERE UN ARRAY JSON VALIDO DI STRINGHE, NON UNA STRINGA SINGOLA.'
    )
    risk_flags: str = dspy.OutputField(
        desc='JSON array of risk flags: [{"flag": str, "severity": "high|medium|low", "clause": str}]'
    )
    non_standard_terms: str = dspy.OutputField(
        desc="JSON array of terms that deviate from client's historical standard"
    )


class ClauseClassification(dspy.Signature):
    """
    Classify a specific contract clause by type, risk level, and business impact.
    """
    clause_text: str = dspy.InputField(desc="The raw clause text to classify")
    contract_context: str = dspy.InputField(desc="Surrounding contract context")

    clause_type: str = dspy.OutputField(
        desc="Type: payment|termination|liability|sla|penalty|renewal|confidentiality|ip|other"
    )
    risk_level: str = dspy.OutputField(desc="Risk level: high|medium|low|none")
    business_impact: str = dspy.OutputField(desc="Brief description of business impact")
    recommended_action: str = dspy.OutputField(desc="Actionable recommendation for this clause")


# ─────────────────────────────────────────────────────────────────────────────
# CLAUSOLA GEMELLA — updated with benchmark reliability
# ─────────────────────────────────────────────────────────────────────────────

class ClausolaGemella(dspy.Signature):
    """
    Compare a client's contract clause against anonymized benchmark data
    from similar companies in the same sector. Provide a percentile ranking
    and specific, actionable negotiation guidance.

    When is_benchmark_available is False, set benchmark_reliability to
    'unavailable' and qualify all outputs accordingly — do not fabricate
    percentiles from empty data.
    """
    clause_text: str = dspy.InputField(desc="The clause to benchmark")
    clause_type: str = dspy.InputField(desc="Type of clause for relevant comparison")
    sector: str = dspy.InputField(desc="Client's industry sector")
    benchmark_corpus: str = dspy.InputField(
        desc="Anonymized benchmark data from similar companies in this sector"
    )
    is_benchmark_available: bool = dspy.InputField(
        desc="False if benchmark data is insufficient (<10 contracts in sector)"
    )

    percentile_ranking: int = dspy.OutputField(
        desc="Percentile 0-100: how favorable is this clause vs sector peers (100=best for client). "
             "Set to -1 if is_benchmark_available is False."
    )
    benchmark_summary: str = dspy.OutputField(
        desc="What similar companies typically have for this clause type"
    )
    deviation_analysis: str = dspy.OutputField(
        desc="How and why this clause deviates from sector standard"
    )
    negotiation_script: str = dspy.OutputField(
        desc="Specific talking points to use when renegotiating this clause"
    )
    priority: str = dspy.OutputField(desc="Renegotiation priority: urgent|high|medium|low")
    benchmark_reliability: str = dspy.OutputField(
        desc="Reliability of this benchmark comparison: high|medium|low|unavailable"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONTRACT SCORE — updated with score history delta
# ─────────────────────────────────────────────────────────────────────────────

class ContractScoring(dspy.Signature):
    """
    Calculate a proprietary ContractIQ Score (0-100) for a contract.
    The score combines risk exposure, revenue quality, client concentration,
    and strategic value into a single actionable number.

    score_history contains previous scores for this contract — use it to
    compute score_delta and provide trend-aware rationale.
    """
    extracted_contract: str = dspy.InputField(desc="JSON of extracted contract fields")
    portfolio_context: str = dspy.InputField(
        desc="Portfolio-level metrics: total ARR, client concentration, avg terms"
    )
    sector_benchmarks: str = dspy.InputField(desc="Sector benchmark data for comparison")
    score_history: str = dspy.InputField(
        desc="JSON array of previous scores [{overall_score, scored_at}], "
             "empty string if this is the first scoring"
    )

    overall_score: int = dspy.OutputField(desc="Overall ContractIQ Score 0-100 (100=best)")
    revenue_quality_score: int = dspy.OutputField(desc="Revenue quality sub-score 0-100")
    risk_exposure_score: int = dspy.OutputField(desc="Risk exposure sub-score 0-100")
    strategic_value_score: int = dspy.OutputField(desc="Strategic/relationship value sub-score 0-100")
    concentration_risk_score: int = dspy.OutputField(desc="Client concentration risk 0-100")
    score_delta: int = dspy.OutputField(
        desc="Delta vs most recent previous overall_score. 0 if this is the first scoring."
    )
    score_rationale: str = dspy.OutputField(desc="Plain-language explanation of the score")
    top_score_drivers: str = dspy.OutputField(
        desc="JSON array of top 3 factors driving the score up or down"
    )
    improvement_actions: str = dspy.OutputField(
        desc="JSON array of specific actions to improve the score at next renewal"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROACTIVE ALERTS
# ─────────────────────────────────────────────────────────────────────────────

class ProactiveAlertGeneration(dspy.Signature):
    """
    Analyze the contract portfolio and generate prioritized, actionable alerts
    for the sales and management team. Alerts should feel like advice from a
    Chief Contract Officer — specific, urgent, and tied to revenue impact.
    """
    portfolio_snapshot: str = dspy.InputField(
        desc="JSON snapshot of all contracts: values, dates, risks, scores"
    )
    historical_patterns: str = dspy.InputField(
        desc="Client-specific historical patterns from RAG (e.g., typical renewal behavior)"
    )
    current_date: str = dspy.InputField(desc="Today's date YYYY-MM-DD")

    critical_alerts: str = dspy.OutputField(
        desc='JSON array of urgent alerts: [{"contract_id": str, "client": str, "alert_type": str, '
             '"message": str, "revenue_at_risk_eur": float, "deadline": str, "action": str}]'
    )
    opportunity_alerts: str = dspy.OutputField(
        desc='JSON array of upsell/renewal opportunities with specific talking points'
    )
    weekly_priorities: str = dspy.OutputField(
        desc="Ordered list of top 5 actions the team should take this week, with owner suggestions"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO RISK ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioRiskAnalysis(dspy.Signature):
    """
    Perform a comprehensive risk analysis on the full contract portfolio.
    Identify systemic risks, concentration issues, and clause-level exposures.
    """
    portfolio_data: str = dspy.InputField(desc="Full portfolio JSON with all contracts")
    client_history: str = dspy.InputField(desc="Historical behavior and payment patterns from RAG")

    concentration_risk: str = dspy.OutputField(
        desc="Analysis of client/revenue concentration risk with Herfindahl index"
    )
    expiry_risk: str = dspy.OutputField(
        desc="Revenue at risk from contracts expiring in next 30/60/90 days"
    )
    clause_risk_summary: str = dspy.OutputField(
        desc="Aggregated clause risks across portfolio with total exposure estimate"
    )
    systemic_risks: str = dspy.OutputField(
        desc="Cross-contract patterns that create systemic risk"
    )
    mitigation_plan: str = dspy.OutputField(
        desc="Prioritized mitigation plan with effort/impact matrix"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RAG-ENHANCED CHAT
# ─────────────────────────────────────────────────────────────────────────────

class ContractQA(dspy.Signature):
    """
    Answer questions about contracts using retrieved context.
    Always cite the exact source clause. Never hallucinate.
    If the answer is not in the provided context, say so explicitly.
    """
    question: str = dspy.InputField(desc="User's question about contracts")
    retrieved_context: str = dspy.InputField(
        desc="Relevant contract chunks retrieved from the client's RAG"
    )
    client_history: str = dspy.InputField(
        desc="Client-specific historical context + injected episodic memory from past interactions"
    )

    answer: str = dspy.OutputField(desc="Direct, actionable answer to the question")
    confidence: str = dspy.OutputField(desc="Confidence level: high|medium|low")
    sources: str = dspy.OutputField(
        desc='JSON array of source citations: [{"contract": str, "clause": str, "text": str}]'
    )
    follow_up_questions: str = dspy.OutputField(
        desc="JSON array of 3 relevant follow-up questions the user might want to ask"
    )


# ─────────────────────────────────────────────────────────────────────────────
# FEEDBACK-DRIVEN OPTIMIZATION (THE TRUE LOCK-IN)
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackQualityMetric(dspy.Signature):
    """
    Evaluate the quality of a ContractIQ response based on user feedback.
    Used as the optimization metric for DSPy MIPROv2 teleprompter.
    This metric drives the per-client prompt optimization that creates lock-in.
    """
    question: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    user_rating: int = dspy.InputField(desc="User rating 1-5")
    user_comment: str = dspy.InputField(desc="Optional user comment on quality")

    quality_score: float = dspy.OutputField(desc="Quality score 0.0-1.0 for optimizer")
    improvement_suggestion: str = dspy.OutputField(desc="How the answer could be improved")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT SUMMARY — Lock-in via strategic information asymmetry
# ─────────────────────────────────────────────────────────────────────────────

class ExportSummary(dspy.Signature):
    """
    Generate an executive summary for PDF/Excel export.
    Deliberately less detailed than the in-platform view.

    Design principle (lock-in):
      Exported data is intentionally high-level — no clause drill-down,
      no score trend charts, no alert timeline. The full intelligence
      lives only inside ContractIQ. Who exports gets a dead snapshot;
      who stays gets a live system.
    """
    portfolio_snapshot: str = dspy.InputField(desc="JSON snapshot portafoglio")
    export_type: str = dspy.InputField(desc="pdf|excel")

    executive_summary: str = dspy.OutputField(
        desc="Max 200 parole, adatto a C-level. Nessun dettaglio clausole."
    )
    key_metrics_table: str = dspy.OutputField(
        desc="JSON array metriche chiave per tabella: "
             '[{"metric": str, "value": str, "unit": str}]'
    )
    top_risks_brief: str = dspy.OutputField(
        desc="Max 3 rischi, formulati ad alto livello, senza citare clausole specifiche"
    )
    cta_message: str = dspy.OutputField(
        desc="Messaggio che invita a vedere i dettagli in piattaforma. "
             "Es: 'Per drill-down completo, score storici e alert in tempo reale, "
             "accedi a ContractIQ.'"
    )