"""
contractiq/dspy_agents/main.py

FastAPI Service for DSPy Agents.
Provides the AI orchestration and handles per-client compilation via MIPROv2.
Routes LLM calls to Regolo AI (Qwen 122B, OpenAI-compatible).
"""

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Optional, List, Union, Any

import dspy
import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field, model_validator, field_validator

from signatures.contract_signatures import ContractQA
from feedback.optimizer import PerClientOptimizer

try:
    import redis.asyncio as redis
except ImportError:
    import redis as redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app_: FastAPI):
    """Application lifespan: startup and shutdown events."""
    global redis_client
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    dspy.settings.configure(lm=ollama_model)  # dspy 2.4.x API
    logger.info("DSPy Agents service ready")
    yield

app = FastAPI(title="ContractIQ DSPy Agents", version="1.0.0", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic model for structured contract extraction (TypedPredictor output)
# ─────────────────────────────────────────────────────────────────────────────

class ProductService(BaseModel):
    name: str = ""
    value_eur: float = 0.0
    recurring: bool = False

    @model_validator(mode="before")
    @classmethod
    def coerce_from_string(cls, v):
        if isinstance(v, str):
            return {"name": v, "value_eur": 0.0, "recurring": False}
        return v

class PenaltyClause(BaseModel):
    trigger: str = ""
    amount_eur: float = 0.0
    type: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_from_string(cls, v):
        if isinstance(v, str):
            return {"trigger": v, "amount_eur": 0.0, "type": "penalty"}
        return v

class RiskFlag(BaseModel):
    flag: str = ""
    severity: str = "medium"
    clause: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_from_string(cls, v):
        if isinstance(v, str):
            return {"flag": v, "severity": "medium", "clause": v}
        return v

class ContractData(BaseModel):
    """All fields extracted from a contract. Every field has a safe default."""
    client_name: str = Field(default="Unknown")
    contract_type: str = Field(default="Agreement")
    total_value_eur: float = Field(default=0.0)
    annual_recurring_revenue: float = Field(default=0.0)
    start_date: str = Field(default="")
    end_date: str = Field(default="")
    auto_renewal: Optional[bool] = None
    renewal_notice_days: int = Field(default=30)
    payment_terms_days: int = Field(default=30)
    discount_percentage: float = Field(default=0.0)
    sla_uptime_percentage: float = Field(default=0.0)
    products_services: List[ProductService] = Field(default_factory=list)
    penalty_clauses: Optional[List[str]] = Field(default_factory=list)
    risk_flags: List[RiskFlag] = Field(default_factory=list)
    non_standard_terms: List[str] = Field(default_factory=list)

    @field_validator('penalty_clauses', mode='before')
    @classmethod
    def string_to_list(cls, v):
        # Se l'LLM manda una stringa, la trasformiamo in una lista di 1 elemento
        if isinstance(v, str):
            return [v]
        # Se manda None, restituiamo una lista vuota
        if v is None:
            return []
        return v

    @model_validator(mode="before")
    @classmethod
    def coerce_nulls(cls, values):
        """Sostituisce None con i default per i campi numerici e stringa."""
        if not isinstance(values, dict):
            return values
        num_fields = {
            "total_value_eur": 0.0,
            "annual_recurring_revenue": 0.0,
            "discount_percentage": 0.0,
            "sla_uptime_percentage": 0.0,
        }
        int_fields = {
            "renewal_notice_days": 30,
            "payment_terms_days": 30,
        }
        str_fields = {
            "start_date": "",
            "end_date": "",
            "client_name": "Unknown",
            "contract_type": "Agreement",
        }
        for f, default in {**num_fields, **int_fields, **str_fields}.items():
            if values.get(f) is None:
                values[f] = default
        return values


class AnalyzeSignature(dspy.Signature):
    """Extract all structured fields from a contract document.
    Return ONLY a valid JSON object (no markdown, no explanation).
    Fields: client_name, contract_type, total_value_eur, annual_recurring_revenue,
    start_date, end_date, auto_renewal, renewal_notice_days, payment_terms_days,
    discount_percentage, sla_uptime_percentage, products_services, penalty_clauses,
    risk_flags, non_standard_terms."""

    contract_text = dspy.InputField(desc="Full text of the contract to analyze")
    client_context = dspy.InputField(desc="Client-specific context: past contracts, preferences")
    extracted_json = dspy.OutputField(desc="Restituisci un JSON puro che inizi con { e finisca con }. Non aggiungere testo prima o dopo. Campi: client_name, contract_type, total_value_eur, annual_recurring_revenue, start_date, end_date, auto_renewal, renewal_notice_days, payment_terms_days, discount_percentage, sla_uptime_percentage, products_services, penalty_clauses, risk_flags, non_standard_terms")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic model for scoring (TypedPredictor output)
# ─────────────────────────────────────────────────────────────────────────────

class ScoringData(BaseModel):
    """Contract scoring results. Every field has a safe default.
    top_score_drivers and improvement_actions are List[str] to avoid
    Pydantic 'Input should be a valid dictionary' errors when Qwen
    sends plain strings instead of nested objects."""
    overall_score: int = Field(default=50, description="Overall ContractIQ Score 0-100 (100=best)")
    revenue_quality_score: int = Field(default=50, description="Revenue quality sub-score 0-100")
    risk_exposure_score: int = Field(default=50, description="Risk exposure sub-score 0-100")
    strategic_value_score: int = Field(default=50, description="Strategic/relationship value sub-score 0-100")
    concentration_risk_score: int = Field(default=50, description="Client concentration risk 0-100")
    score_rationale: str = Field(default="", description="Plain-language explanation of the score")
    top_score_drivers: List[str] = Field(default_factory=list, description="Top 3 factors driving the score, as plain strings")
    improvement_actions: List[str] = Field(default_factory=list, description="Actions to improve at next renewal, as plain strings")


class ScoringSignature(dspy.Signature):
    """Calculate a proprietary ContractIQ Score (0-100) for a contract.
    Return ONLY a valid JSON object (no markdown, no explanation).
    Fields: overall_score, revenue_quality_score, risk_exposure_score,
    strategic_value_score, concentration_risk_score, score_rationale,
    top_score_drivers, improvement_actions."""

    extracted_contract = dspy.InputField(desc="JSON of extracted contract fields")
    portfolio_context = dspy.InputField(desc="Portfolio-level metrics: total ARR, client concentration, avg terms")
    sector_benchmarks = dspy.InputField(desc="Sector benchmark data for comparison")
    scoring_json = dspy.OutputField(desc="Restituisci un JSON puro che inizi con { e finisca con }. Non aggiungere testo prima o dopo. Campi: overall_score, revenue_quality_score, risk_exposure_score, strategic_value_score, concentration_risk_score, score_rationale, top_score_drivers, improvement_actions")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic model for ClausolaGemella (TypedPredictor output)
# ─────────────────────────────────────────────────────────────────────────────

class GemellaData(BaseModel):
    """Clausola Gemella benchmark result. Every field has a safe default."""
    percentile_ranking: int = Field(default=50, description="Percentile 0-100: how favorable vs sector peers (100=best)")
    benchmark_summary: str = Field(default="", description="What similar companies typically have for this clause type")
    deviation_analysis: str = Field(default="", description="How and why this clause deviates from sector standard")
    negotiation_script: str = Field(default="", description="Specific talking points for renegotiating this clause")
    priority: str = Field(default="medium", description="Renegotiation priority: urgent|high|medium|low")


class GemellaSignature(dspy.Signature):
    """Compare a contract clause against anonymized benchmark data
    from similar companies in the same sector.
    Return ONLY a valid JSON object (no markdown, no explanation).
    Fields: percentile_ranking, benchmark_summary, deviation_analysis,
    negotiation_script, priority."""

    clause_text = dspy.InputField(desc="The clause to benchmark")
    clause_type = dspy.InputField(desc="Type of clause for relevant comparison")
    sector = dspy.InputField(desc="Client's industry sector")
    benchmark_corpus = dspy.InputField(desc="Anonymized benchmark data from similar companies in this sector")
    gemella_json = dspy.OutputField(desc="Restituisci un JSON puro che inizi con { e finisca con }. Non aggiungere testo prima o dopo. Campi: percentile_ranking, benchmark_summary, deviation_analysis, negotiation_script, priority")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

RAG_URL = os.getenv("RAG_SERVICE_URL", "http://rag_service:8002")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
BENCHMARK_SERVICE_URL = os.getenv("BENCHMARK_SERVICE_URL", "http://benchmark_service:8006")
EXTERNAL_LLM_URL = os.getenv("EXTERNAL_LLM_URL")       # defined in .env
EXTERNAL_LLM_MODEL = os.getenv("EXTERNAL_LLM_MODEL", "qwen3.5-122b")  # default per Regolo AI
EXTERNAL_LLM_API_KEY = os.getenv("EXTERNAL_LLM_API_KEY", "")  # optional: bearer token

if not EXTERNAL_LLM_URL:
    logger.warning("EXTERNAL_LLM_URL not set — using localhost fallback")
if not os.getenv("EXTERNAL_LLM_MODEL"):
    logger.warning(f"EXTERNAL_LLM_MODEL not set — using default: {EXTERNAL_LLM_MODEL}")

# Configurazione per provider OpenAI-compatible (Regolo AI)
# Assicura che api_base termini con "/" per evitare errori di concatenazione (es. v1chat)
_api_base = EXTERNAL_LLM_URL.rstrip("/") + "/" if EXTERNAL_LLM_URL else "http://localhost:11434/v1/"
ollama_model = dspy.OpenAI(
    model=EXTERNAL_LLM_MODEL,
    api_key=EXTERNAL_LLM_API_KEY,
    api_base=_api_base,
    model_type="chat",
    max_tokens=8192,
    temperature=0.2,
    reasoning_effort=None,
)
logger.info(f"DSPy configured with Regolo AI: {EXTERNAL_LLM_MODEL} at {_api_base}")

redis_client = None
optimizer = PerClientOptimizer()


def extract_json_from_response(text) -> dict:
    """
    Estrae un JSON object dalla risposta LLM, gestendo:
    - risposta None o vuota
    - markdown code blocks (```json ... ```)
    - testo extra prima/dopo il JSON
    """
    if not text or not isinstance(text, str):
        return {}

    text = text.strip()

    # 1. Parse diretto
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Estrai da code block markdown
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Trova il primo { ... } block
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    return {}


def fallback_extract_from_history() -> dict:
    """
    Fallback: recupera l'ultima risposta grezza da ollama_model.history
    e prova a estrarre JSON con regex { ... }.
    """
    try:
        history = getattr(ollama_model, "history", None)
        if not history:
            return {}
        # history è una lista di dict, l'ultimo contiene la risposta
        last = history[-1] if isinstance(history, list) else None
        if not last:
            return {}
        # Prova vari formati di history di dspy.OpenAI
        raw = None
        if isinstance(last, dict):
            raw = last.get("response", last.get("output", ""))
            if isinstance(raw, dict):
                raw = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        elif isinstance(last, str):
            raw = last
        if raw:
            return extract_json_from_response(raw)
    except Exception as e:
        logger.debug(f"History fallback failed: {e}")
    return {}


# Startup gestito dal lifespan context manager (vedi sopra).


@app.get("/health")
async def health():
    """Healthcheck endpoint for Docker."""
    return {"status": "ok", "service": "dspy_agents"}


# ─────────────────────────────────────────────────────────────────────────────
# DSPy Modules
# ─────────────────────────────────────────────────────────────────────────────

class ContractOrchestrator(dspy.Module):
    """The central agent optimized per client."""
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(ContractQA)

    def forward(self, question, retrieved_context, client_history="None"):
        return self.qa(question=question, retrieved_context=retrieved_context, client_history=client_history)


base_orchestrator = ContractOrchestrator()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Call RAG Service
# ─────────────────────────────────────────────────────────────────────────────

async def get_rag_context(client_id: str, question: str) -> str:
    """Query the RAG service for context."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{RAG_URL}/retrieve",
                json={"client_id": client_id, "query": question, "top_k": 5}
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("context", "")
    except Exception as e:
        logger.error(f"Failed to fetch RAG context: {e}")
        return "N/A"


async def get_rag_history(client_id: str) -> str:
    return "N/A"  # Simplified for MVP




# ─────────────────────────────────────────────────────────────────────────────
# Analysis Endpoints (Upload)
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    client_id: str
    contract_text: str
    sector: str


@app.post("/analyze/contract")
async def analyze_contract(req: AnalyzeRequest):
    """
    Extract data, score it, and find twin clauses.
    Use DSPy Predictors for all steps to avoid static prompts.
    """
    contract_trunc = req.contract_text[:8000]

    # Initialize DSPy Predictors
    extract_predictor = dspy.Predict(AnalyzeSignature)
    score_predictor = dspy.Predict(ScoringSignature)
    gemella_predictor = dspy.Predict(GemellaSignature)

    # ── 1. Extraction ────────────────────────────────────────────────────────
    extraction = ContractData().model_dump()  # safe default
    try:
        pred = extract_predictor(contract_text=contract_trunc, client_context="")
        raw = getattr(pred, "extracted_json", "")
        parsed = extract_json_from_response(raw)
        if parsed:
            # --- INIZIO SCUDO LISTE ---
            # Definisci tutti i campi che il tuo Pydantic si aspetta come lista
            list_fields = [
                "penalty_clauses", 
                "products_services", 
                "risk_flags", 
                "non_standard_terms"
            ]

            for field in list_fields:
                if field in parsed and isinstance(parsed[field], str):
                    # Se l'LLM ha mandato una stringa, proviamo a dividerla per virgole
                    if "," in parsed[field]:
                        parsed[field] = [item.strip() for item in parsed[field].split(",")]
                    else:
                        # Altrimenti la infiliamo nuda e cruda in una lista
                        parsed[field] = [parsed[field]]
            # --- FINE SCUDO LISTE ---
            
            extraction = ContractData(**parsed).model_dump()
            logger.info(f"Extraction OK: {extraction.get('client_name', 'Unknown')}")
        else:
            logger.warning(f"Extraction: DSPy response unparseable. Raw: {raw[:200]}")
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)

    # ── 2. Scoring ───────────────────────────────────────────────────────────
    scoring_data = ScoringData()  # safe default
    
    # Chiama benchmark service per sector_benchmarks
    sector_benchmarks_str = ""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            deal_size = extraction.get("total_value_eur", 0)
            resp = await client.get(
                f"{BENCHMARK_SERVICE_URL}/benchmark/summary",
                params={
                    "sector": req.sector,
                    "deal_size": deal_size,
                    "country": "IT"
                }
            )
            if resp.status_code == 200:
                sector_benchmarks_str = resp.json().get("summary_text", "")
    except Exception as e:
        logger.warning(f"Benchmark service unavailable, proceeding without: {e}")

    try:
        pred = score_predictor(
            extracted_contract=json.dumps(extraction, ensure_ascii=False),
            portfolio_context="",
            sector_benchmarks=sector_benchmarks_str,
        )
        raw = getattr(pred, "scoring_json", "")
        parsed = extract_json_from_response(raw)
        if parsed:
            scoring_data = ScoringData(**parsed)
        else:
            logger.warning(f"Scoring: DSPy response unparseable. Raw: {raw[:200]}")
    except Exception as e:
        logger.error(f"Scoring failed: {e}", exc_info=True)

    scoring = {
        "overall_score": scoring_data.overall_score,
        "revenue_quality_score": scoring_data.revenue_quality_score,
        "risk_exposure_score": scoring_data.risk_exposure_score,
        "strategic_value_score": scoring_data.strategic_value_score,
        "concentration_risk_score": scoring_data.concentration_risk_score,
    }

    # ── 3. Clausola Gemella ──────────────────────────────────────────────────
    gemella_out = []
    risks = extraction.get("risk_flags", [])
    if risks:
        first_risk = risks[0] if isinstance(risks, list) and risks else {}
        clause = first_risk.get("clause", "") if isinstance(first_risk, dict) else str(risks)
        # Benchmark specifico per clausola
        benchmark_corpus_str = ""
        clause_type_val = first_risk.get("flag", "Liability") if isinstance(first_risk, dict) else "Liability"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{BENCHMARK_SERVICE_URL}/benchmark/clause",
                    params={
                        "clause_type": clause_type_val,
                        "sector": req.sector
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("sample_size", 0) >= 5:
                        benchmark_corpus_str = (
                            f"Nel tuo settore ({req.sector}), questa clausola ha valore mediano "
                            f"{data.get('p50_value', 'N/A')} con range P25-P75 di {data.get('p25_value', 'N/A')}-{data.get('p75_value', 'N/A')}. "
                            f"Il tuo contratto è al {data.get('percentile_rank', 50)}° percentile su {data['sample_size']} contratti."
                        )
        except Exception as e:
            logger.warning(f"Clause benchmark unavailable: {e}")

        try:
            pred = gemella_predictor(
                clause_text=clause or "N/A",
                clause_type=clause_type_val,
                sector=req.sector,
                benchmark_corpus=benchmark_corpus_str,
            )
            raw = getattr(pred, "gemella_json", "")
            parsed = extract_json_from_response(raw)
            if parsed:
                gemella_out.append(GemellaData(**parsed).model_dump())
            else:
                logger.warning(f"Gemella: DSPy response unparseable. Raw: {raw[:200]}")
        except Exception as e:
            logger.error(f"ClausolaGemella failed: {e}", exc_info=True)

    # ── 4. Ingest Benchmark Data ─────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(
                f"{BENCHMARK_SERVICE_URL}/benchmark/ingest",
                json={
                    "client_id": req.client_id,
                    "sector": req.sector,
                    "extraction": extraction
                }
            )
    except Exception as e:
        logger.debug(f"Failed to post to benchmark ingest: {e}")

    # ── Response ─────────────────────────────────────────────────────────────
    return {
        "analysis": {
            "extraction": extraction,
            "scoring": scoring,
            "clausola_gemella": gemella_out,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Q&A Endpoint
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    client_id: str
    session_id: str
    question: str


@app.post("/chat/question")
async def chat_question(req: ChatRequest):
    """
    Answer a question using the RAG index and the optimized DSPy program specific
    to THIS client.
    """
    # Load optimized version if exists
    active_agent = optimizer.load_optimized_orchestrator(req.client_id, base_orchestrator)
    
    # Get Context
    context = await get_rag_context(req.client_id, req.question)
    
    # Generate Answer
    prediction = active_agent(question=req.question, retrieved_context=context)
    
    # Safely parse sources
    def parse_blocks(raw_val, default):
        try:
            return json.loads(raw_val) if isinstance(raw_val, str) else default
        except (json.JSONDecodeError, ValueError, TypeError):
            return default

    return {
        "response": {
            "answer": getattr(prediction, "answer", "N/A"),
            "confidence": getattr(prediction, "confidence", "medium"),
            "sources": parse_blocks(getattr(prediction, "sources", "[]"), []),
            "follow_up_questions": parse_blocks(getattr(prediction, "follow_up_questions", "[]"), []),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feedback & Optimization (LOCK-IN Engine)
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    client_id: str
    session_id: str
    question: str
    answer: str
    sources: Union[list, str]
    user_rating: int
    user_comment: Optional[str] = None


async def trigger_mipro_optimization(client_id: str):
    """Background task to run optimization when threshold is met."""
    if not redis_client:
        return
        
    feedback_key = f"feedback:{client_id}"
    raw_data = await redis_client.lrange(feedback_key, 0, -1)
    
    examples = []
    for item in raw_data:
        data = json.loads(item)
        examples.append(
            dspy.Example(
                question=data["question"],
                predicted_answer=data["answer"],
                user_rating=data["user_rating"],
                user_comment=data.get("user_comment", "")
            ).with_inputs('question')
        )
        
    if len(examples) >= 20:
        logger.info(f"Threshold reached ({len(examples)}/20). Triggering MIPROv2 for {client_id}.")
        # Run optimization (blocking operation in thread pool normally, simplified here)
        optimizer.run_optimization(client_id, base_orchestrator, examples)
        

@app.post("/feedback")
async def receive_feedback(req: FeedbackRequest, background_tasks: BackgroundTasks):
    """Store feedback as a training example for the DSPy optimizer."""
    if not redis_client:
        return {"status": "redis_unavailable"}
        
    feedback_key = f"feedback:{req.client_id}"
    data = {
        "question": req.question,
        "answer": req.answer,
        "user_rating": req.user_rating,
        "user_comment": req.user_comment or ""
    }
    
    await redis_client.rpush(feedback_key, json.dumps(data))
    count = await redis_client.llen(feedback_key)
    
    # Check if we should optimize
    if count >= 20 and count % 20 == 0:  # Re-optimize every 20 new examples
        background_tasks.add_task(trigger_mipro_optimization, req.client_id)
        
    return {
        "status": "stored",
        "current_examples": count,
        "threshold": 20,
        "optimizing_now": count >= 20 and count % 20 == 0
    }


@app.get("/optimization/status/{client_id}")
async def get_optimization_status(client_id: str):
    """Check if this client has an optimized agent."""
    status = optimizer.get_optimization_status(client_id)
    if redis_client:
        count = await redis_client.llen(f"feedback:{client_id}")
        status["current_feedback_count"] = count
    return status

