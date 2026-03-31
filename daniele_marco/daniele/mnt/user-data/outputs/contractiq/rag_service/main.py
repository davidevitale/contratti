# contractiq/rag_service/main.py
"""
Per-Client RAG Service.

Architecture:
  - Each client gets their own ChromaDB COLLECTION: client_{client_id}_contracts
  - Each client also has a benchmark collection: benchmark_{sector}
  - Clausola Gemella queries the benchmark collection for cross-client comparison

Lock-in mechanisms:
  - Every contract ingested builds the client's private corpus
  - Every user interaction is stored in episodic memory and injected as context
  - ContractIQ Score history persisted in PostgreSQL — irreproducible time series
  - The richer the corpus, the more precise retrieval → better answers
  - Deleting the account means losing years of structured contract history + score trends

Chunking strategy:
  - NOT fixed-size chunks (naive approach)
  - Clause-level semantic chunking: each clause is an independent unit
  - Each chunk carries rich metadata for filtering

Embeddings:
  - SentenceTransformer all-MiniLM-L6-v2 — free, local, no API key required
  - ChromaDB handles embedding automatically via embedding_function on each collection
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ContractIQ RAG Service", version="2.0.0")

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql+asyncpg://contractiq:contractiq_dev_password@postgres:5432/contractiq",
)
BENCHMARK_MIN_SAMPLE = int(os.getenv("BENCHMARK_MIN_SAMPLE", "10"))
EPISODIC_MIN_RATING = int(os.getenv("EPISODIC_MIN_RATING", "3"))

# Shared embedding function — all-MiniLM-L6-v2 is free and runs locally
_EMBED_FN = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Sentiment word lists (IT + EN)
_POSITIVE_WORDS = {
    "ottimo", "perfetto", "utile", "preciso", "corretto", "eccellente",
    "great", "perfect", "correct", "useful", "accurate", "excellent", "good",
}
_NEGATIVE_WORDS = {
    "sbagliato", "errato", "inutile", "pessimo", "incorreto",
    "wrong", "incorrect", "useless", "bad", "terrible", "poor",
}


# ── ChromaDB ──────────────────────────────────────────────────────────────────

chroma_client: Optional[chromadb.AsyncHttpClient] = None  # BUG FIX: no await


async def get_chroma() -> chromadb.AsyncHttpClient:
    global chroma_client
    if chroma_client is None:
        chroma_client = await chromadb.AsyncHttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return chroma_client


def client_collection_name(client_id: str) -> str:
    return f"client_{client_id}_contracts"


def benchmark_collection_name(sector: str) -> str:
    return f"benchmark_{sector.lower().replace(' ', '_')}"


def episodic_collection_name(client_id: str) -> str:
    return f"episodic_{client_id}"


# ── PostgreSQL ────────────────────────────────────────────────────────────────

engine = create_async_engine(POSTGRES_URL, echo=False)


async def get_db_conn():
    async with engine.connect() as conn:
        yield conn


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_sentiment(comment: str) -> str:
    tokens = set(comment.lower().split())
    if tokens & _POSITIVE_WORDS:
        return "positive"
    if tokens & _NEGATIVE_WORDS:
        return "negative"
    return "neutral"


def _compute_trend(scores: list[int]) -> str:
    """
    Compare mean of last 3 scores vs mean of preceding scores.
    Returns 'improving' | 'stable' | 'declining'.
    """
    if len(scores) < 2:
        return "stable"
    recent = scores[-3:]
    prior = scores[:-3] if len(scores) > 3 else scores[:1]
    delta = (sum(recent) / len(recent)) - (sum(prior) / len(prior))
    if delta > 5:
        return "improving"
    if delta < -5:
        return "declining"
    return "stable"


# ── Clause-Level Semantic Chunker ─────────────────────────────────────────────

def semantic_clause_chunker(text: str, contract_metadata: dict) -> list[dict]:
    """
    Split contract text into clause-level chunks.

    Strategy:
      1. Detect numbered clauses (1., 2.1, Art. 1, ARTICLE I, etc.)
      2. Detect standard section headers
      3. Each clause becomes one chunk with rich metadata
      4. Fallback: 500-token sliding window with 100-token overlap
    """
    chunks = []
    clause_patterns = [
        r'\n\s*(?:Art(?:icle|icolo)?\.?\s*\d+[\.\d]*)',
        r'\n\s*\d+[\.\d]+\s+[A-Z]',
        r'\n\s*(?:CLAUSE|CLAUSOLA|SEZIONE|SECTION)\s+\d+',
        r'\n\s*(?:Whereas|WHEREAS|Premesso che)',
    ]

    combined_pattern = '|'.join(f'({p})' for p in clause_patterns)
    splits = re.split(combined_pattern, text, flags=re.IGNORECASE)

    if len(splits) > 3:
        current_clause = ""
        clause_num = 0
        for part in splits:
            # BUG FIX: explicit None check before calling .strip()
            if part is not None and len(part.strip()) > 20:
                if re.match(combined_pattern, part, re.IGNORECASE):
                    if current_clause.strip():
                        chunks.append({
                            "text": current_clause.strip(),
                            "metadata": {
                                **contract_metadata,
                                "chunk_type": "clause",
                                "clause_index": clause_num,
                                "char_count": len(current_clause),
                            },
                        })
                        clause_num += 1
                    current_clause = part
                else:
                    current_clause += " " + part
        if current_clause.strip():
            chunks.append({
                "text": current_clause.strip(),
                "metadata": {
                    **contract_metadata,
                    "chunk_type": "clause",
                    "clause_index": clause_num,
                },
            })
    else:
        words = text.split()
        chunk_size = 400
        overlap = 80
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = " ".join(words[i:i + chunk_size])
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **contract_metadata,
                        "chunk_type": "window",
                        "chunk_index": i // (chunk_size - overlap),
                    },
                })

    return chunks


# ── Request / Response Models ─────────────────────────────────────────────────

class IngestRequest(BaseModel):
    client_id: str
    contract_id: str
    contract_text: str
    metadata: dict


class RetrieveRequest(BaseModel):
    client_id: str
    query: str
    top_k: int = 5
    filter_metadata: Optional[dict] = None


class BenchmarkIngestRequest(BaseModel):
    sector: str
    clause_text: str
    clause_type: str
    metadata: dict


class EpisodicMemoryRequest(BaseModel):
    client_id: str
    question: str
    answer: str
    rating: int


class ChatMemoryRequest(BaseModel):
    client_id: str
    session_id: str
    message: str


class ScoreHistoryRequest(BaseModel):
    client_id: str
    contract_id: str
    score_data: dict
    scored_at: Optional[str] = None  # ISO datetime string; defaults to NOW()


class UserFeedbackRequest(BaseModel):
    client_id: str
    session_id: str
    question: str
    answer: str
    rating: int
    comment: str = ""
    context_tags: list[str] = []


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "rag_service", "version": "2.0.0"}


# ── Contract Ingestion ────────────────────────────────────────────────────────

@app.post("/ingest/contract")
async def ingest_contract(req: IngestRequest) -> dict:
    """
    Ingest a contract into the client's isolated RAG collection.
    Performs semantic clause-level chunking; ChromaDB embeds automatically.
    """
    chroma = await get_chroma()
    collection = await chroma.get_or_create_collection(
        name=client_collection_name(req.client_id),
        metadata={"hnsw:space": "cosine"},
        embedding_function=_EMBED_FN,
    )

    chunks = semantic_clause_chunker(req.contract_text, {
        **req.metadata,
        "contract_id": req.contract_id,
        "client_id": req.client_id,
    })

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks extracted from contract")

    texts = [c["text"] for c in chunks]
    ids = [
        f"{req.contract_id}_chunk_{i}_{hashlib.md5(c['text'].encode()).hexdigest()[:8]}"
        for i, c in enumerate(chunks)
    ]

    # ChromaDB handles embeddings automatically — no embeddings= param
    await collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
    )

    logger.info(
        "Ingested %d chunks for client=%s contract=%s",
        len(chunks), req.client_id, req.contract_id,
    )
    return {"status": "ingested", "chunks": len(chunks), "contract_id": req.contract_id}


# ── Retrieval ─────────────────────────────────────────────────────────────────

@app.post("/retrieve")
async def retrieve(req: RetrieveRequest) -> dict:
    """
    Retrieve relevant contract chunks for a client query.
    Also returns client history from episodic memory.
    """
    chroma = await get_chroma()

    context = ""
    try:
        collection = await chroma.get_collection(
            client_collection_name(req.client_id),
            embedding_function=_EMBED_FN,
        )
        results = await collection.query(
            query_texts=[req.query],  # ChromaDB embeds the query automatically
            n_results=req.top_k,
            where=req.filter_metadata or None,
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        context_parts = []
        for doc, meta in zip(documents, metadatas):
            context_parts.append(
                f"[Contract: {meta.get('contract_id', 'unknown')} | "
                f"Client: {meta.get('client_name', 'unknown')} | "
                f"Clause #{meta.get('clause_index', '?')}]\n{doc}"
            )
        context = "\n\n---\n\n".join(context_parts)
    except Exception as exc:
        logger.warning("Contract retrieval failed for client=%s: %s", req.client_id, exc)

    client_history = ""
    try:
        episodic = await chroma.get_collection(
            episodic_collection_name(req.client_id),
            embedding_function=_EMBED_FN,
        )
        ep_results = await episodic.query(query_texts=[req.query], n_results=3)
        ep_docs = ep_results.get("documents", [[]])[0]
        if ep_docs:
            client_history = "Past interactions:\n" + "\n".join(ep_docs)
    except Exception:
        pass  # episodic collection may not exist yet — silent

    return {"context": context, "client_history": client_history}


# ── Benchmark ─────────────────────────────────────────────────────────────────

@app.get("/benchmarks/{sector}")
async def get_benchmarks(sector: str) -> dict:
    """
    Return anonymized benchmark data for Clausola Gemella.
    Includes is_benchmark_available flag — never returns hardcoded data silently.
    """
    chroma = await get_chroma()
    try:
        collection = await chroma.get_collection(
            benchmark_collection_name(sector),
            embedding_function=_EMBED_FN,
        )
        results = await collection.get(limit=50, include=["documents", "metadatas"])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        sla_values = [m.get("sla_uptime") for m in metas if m.get("sla_uptime")]
        payment_terms = [m.get("payment_days") for m in metas if m.get("payment_days")]
        sample_size = len(docs)
        is_available = sample_size >= BENCHMARK_MIN_SAMPLE

        response = {
            "sector": sector,
            "sample_size": sample_size,
            "is_benchmark_available": is_available,
            "benchmarks": {
                "avg_sla_uptime": (
                    sum(sla_values) / len(sla_values) if sla_values else None
                ),
                "avg_payment_days": (
                    sum(payment_terms) / len(payment_terms) if payment_terms else None
                ),
                "corpus_doc_count": sample_size,
            },
            "corpus_summary": "\n".join(docs[:10]),
        }
        if not is_available:
            response["warning"] = (
                f"Dati insufficienti per benchmark affidabile "
                f"(<{BENCHMARK_MIN_SAMPLE} contratti nel settore)"
            )
        return response

    except Exception as exc:
        logger.warning("Benchmark retrieval failed for sector=%s: %s", sector, exc)
        return {
            "sector": sector,
            "sample_size": 0,
            "is_benchmark_available": False,
            "benchmarks": {"avg_sla_uptime": None, "avg_payment_days": None},
            "corpus_summary": "",
            "warning": "Nessun dato benchmark disponibile per questo settore.",
        }


@app.post("/ingest/benchmark")
async def ingest_benchmark(req: BenchmarkIngestRequest) -> dict:
    """
    Add an anonymized clause to the sector benchmark corpus.
    Called automatically after each contract ingestion (anonymized).
    Powers Clausola Gemella.
    """
    chroma = await get_chroma()
    collection = await chroma.get_or_create_collection(
        benchmark_collection_name(req.sector),
        embedding_function=_EMBED_FN,
    )

    # BUG FIX: include sector + clause_type in hash to prevent cross-sector collisions
    chunk_id = hashlib.md5(
        f"{req.sector}:{req.clause_type}:{req.clause_text}".encode()
    ).hexdigest()

    await collection.upsert(
        ids=[chunk_id],
        documents=[req.clause_text],
        metadatas=[{**req.metadata, "sector": req.sector, "clause_type": req.clause_type}],
    )
    return {"status": "benchmark_ingested"}


# ── Episodic Memory (legacy endpoint — kept for compatibility) ─────────────────

@app.post("/episodic/store")
async def store_episodic(req: EpisodicMemoryRequest) -> dict:
    """
    Store a Q&A interaction in the client's episodic memory.
    High-rated interactions (>= EPISODIC_MIN_RATING) get priority storage.
    """
    if req.rating < EPISODIC_MIN_RATING:
        return {"status": "skipped_low_rating"}

    chroma = await get_chroma()
    collection = await chroma.get_or_create_collection(
        episodic_collection_name(req.client_id),
        embedding_function=_EMBED_FN,
    )

    memory_text = f"Q: {req.question}\nA: {req.answer}"
    memory_id = hashlib.md5(memory_text.encode()).hexdigest()

    await collection.upsert(
        ids=[memory_id],
        documents=[memory_text],
        metadatas=[{"rating": req.rating, "timestamp": datetime.now(timezone.utc).isoformat()}],
    )
    return {"status": "stored"}


@app.post("/chat/memory/store")
async def store_chat_memory(req: ChatMemoryRequest) -> dict:
    """Store raw user chat interactions in episodic memory for future retrieval."""
    try:
        chroma = await get_chroma()
        collection = await chroma.get_or_create_collection(
            episodic_collection_name(req.client_id),
            embedding_function=_EMBED_FN,
        )

        memory_id = hashlib.md5(f"{req.session_id}:{req.message}".encode()).hexdigest()
        
        await collection.upsert(
            ids=[memory_id],
            documents=[f"USER QUESTION: {req.message}"],
            metadatas=[{
                "session_id": req.session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "raw_chat",
            }],
        )
        return {"status": "stored"}
    except Exception as exc:
        logger.warning("Chat memory store failed for client=%s: %s", req.client_id, exc)
        return {"status": "error", "detail": str(exc)}


# ── Score History — Lock-in #1 ────────────────────────────────────────────────

@app.post("/score/history/store")
async def store_score_history(req: ScoreHistoryRequest) -> dict:
    """
    Persist a ContractIQ Score snapshot to PostgreSQL.
    Builds the proprietary time-series that becomes the client's irreplaceable asset.
    """
    scored_at = req.scored_at or datetime.now(timezone.utc).isoformat()

    stmt = sa.text("""
        INSERT INTO contract_score_history
            (client_id, contract_id, overall_score, revenue_quality,
             risk_exposure, strategic_value, concentration_risk,
             score_data, scored_at)
        VALUES
            (:client_id, :contract_id, :overall_score, :revenue_quality,
             :risk_exposure, :strategic_value, :concentration_risk,
             :score_data, :scored_at)
        RETURNING id
    """)

    params = {
        "client_id": req.client_id,
        "contract_id": req.contract_id,
        "overall_score": req.score_data.get("overall_score", 0),
        "revenue_quality": req.score_data.get("revenue_quality"),
        "risk_exposure": req.score_data.get("risk_exposure"),
        "strategic_value": req.score_data.get("strategic_value"),
        "concentration_risk": req.score_data.get("concentration_risk"),
        "score_data": json.dumps(req.score_data),
        "scored_at": scored_at,
    }

    try:
        async with engine.connect() as conn:
            result = await conn.execute(stmt, params)
            await conn.commit()
            row_id = result.scalar_one()
        logger.info("Stored score history id=%d client=%s contract=%s", row_id, req.client_id, req.contract_id)
        return {"status": "stored", "id": row_id}
    except Exception as exc:
        logger.error("Failed to store score history for client=%s: %s", req.client_id, exc)
        raise HTTPException(status_code=500, detail="Score history storage failed")


@app.get("/score/history/{client_id}")
async def get_score_history(
    client_id: str,
    contract_id: Optional[str] = Query(default=None),
    from_date: Optional[str] = Query(default=None),
    to_date: Optional[str] = Query(default=None),
) -> dict:
    """
    Retrieve ContractIQ Score history for a client.
    Returns trend signal: improving | stable | declining.
    This endpoint powers the lock-in dashboard showing score evolution over time.
    """
    filters = ["client_id = :client_id"]
    params: dict = {"client_id": client_id}

    if contract_id:
        filters.append("contract_id = :contract_id")
        params["contract_id"] = contract_id
    if from_date:
        filters.append("scored_at >= :from_date")
        params["from_date"] = from_date
    if to_date:
        filters.append("scored_at <= :to_date")
        params["to_date"] = to_date

    where_clause = " AND ".join(filters)
    stmt = sa.text(f"""
        SELECT id, client_id, contract_id, overall_score, revenue_quality,
               risk_exposure, strategic_value, concentration_risk,
               score_data, scored_at, created_at
        FROM contract_score_history
        WHERE {where_clause}
        ORDER BY scored_at ASC
    """)

    try:
        async with engine.connect() as conn:
            result = await conn.execute(stmt, params)
            rows = result.mappings().all()

        history = [dict(row) for row in rows]
        # Serialize datetime fields
        for row in history:
            for k, v in row.items():
                if hasattr(v, "isoformat"):
                    row[k] = v.isoformat()
            if isinstance(row.get("score_data"), str):
                try:
                    row["score_data"] = json.loads(row["score_data"])
                except (json.JSONDecodeError, TypeError):
                    pass

        overall_scores = [r["overall_score"] for r in history]
        trend = _compute_trend(overall_scores)

        return {"history": history, "count": len(history), "trend": trend}
    except Exception as exc:
        logger.error("Failed to retrieve score history for client=%s: %s", client_id, exc)
        raise HTTPException(status_code=500, detail="Score history retrieval failed")


# ── User Feedback — Lock-in #2 (interaction memory) ──────────────────────────

@app.post("/user/feedback/store")
async def store_user_feedback(req: UserFeedbackRequest) -> dict:
    """
    Store a user interaction with rating and optional comment.

    Two-layer storage:
      1. ChromaDB episodic collection (if rating >= EPISODIC_MIN_RATING)
         → used as prompt injection context in future QA calls
      2. PostgreSQL user_feedback table (always)
         → powers analytics, MIPROv2 training data, lock-in reporting

    The comment_sentiment is derived automatically from keyword matching (IT + EN).
    """
    sentiment = _detect_sentiment(req.comment)
    episodic_stored = False

    # Layer 1: ChromaDB episodic memory
    if req.rating >= EPISODIC_MIN_RATING:
        try:
            chroma = await get_chroma()
            collection = await chroma.get_or_create_collection(
                episodic_collection_name(req.client_id),
                embedding_function=_EMBED_FN,
            )
            tags_str = ",".join(req.context_tags) if req.context_tags else ""
            document = f"Q: {req.question}\nA: {req.answer}\nTags: {tags_str}"
            memory_id = hashlib.md5(
                f"{req.session_id}:{req.question}".encode()
            ).hexdigest()

            await collection.upsert(
                ids=[memory_id],
                documents=[document],
                metadatas=[{
                    "rating": req.rating,
                    "comment": req.comment,
                    "session_id": req.session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "comment_sentiment": sentiment,
                    "context_tags": tags_str,
                }],
            )
            episodic_stored = True
            logger.info(
                "Episodic memory stored for client=%s session=%s rating=%d",
                req.client_id, req.session_id, req.rating,
            )
        except Exception as exc:
            logger.warning(
                "Episodic storage failed for client=%s: %s", req.client_id, exc
            )

    # Layer 2: PostgreSQL
    stmt = sa.text("""
        INSERT INTO user_feedback
            (client_id, session_id, question, answer, rating,
             comment, context_tags, comment_sentiment)
        VALUES
            (:client_id, :session_id, :question, :answer, :rating,
             :comment, :context_tags, :comment_sentiment)
    """)
    try:
        async with engine.connect() as conn:
            await conn.execute(stmt, {
                "client_id": req.client_id,
                "session_id": req.session_id,
                "question": req.question,
                "answer": req.answer,
                "rating": req.rating,
                "comment": req.comment,
                "context_tags": json.dumps(req.context_tags),
                "comment_sentiment": sentiment,
            })
            await conn.commit()
    except Exception as exc:
        logger.error(
            "PostgreSQL feedback storage failed for client=%s: %s", req.client_id, exc
        )

    status = "stored" if req.rating >= EPISODIC_MIN_RATING else "skipped_low_rating"
    return {"status": status, "episodic_stored": episodic_stored}


@app.get("/user/feedback/context/{client_id}")
async def get_feedback_context(
    client_id: str,
    query: str = Query(..., description="Query to retrieve relevant past interactions"),
    top_k: int = Query(default=3, ge=1, le=10),
) -> dict:
    """
    Retrieve relevant past user interactions as a prompt injection string.

    Called by the gateway before every QA request.
    The returned injection_context is prepended to client_history,
    giving the LLM awareness of this user's preferences and past answers.
    This is the interaction-level lock-in: the more a user interacts,
    the more personalized and accurate the answers become.
    """
    try:
        chroma = await get_chroma()
        collection = await chroma.get_collection(
            episodic_collection_name(client_id),
            embedding_function=_EMBED_FN,
        )
        results = await collection.query(query_texts=[query], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        if not docs:
            return {"injection_context": "", "sources_count": 0}

        lines = []
        for doc, meta in zip(docs, metas):
            rating = meta.get("rating", "?")
            tags = meta.get("context_tags", "")
            lines.append(f"[Rating:{rating} | Tag:{tags}] {doc}")

        injection_context = "CONTESTO STORICO UTENTE:\n" + "\n\n".join(lines)
        return {"injection_context": injection_context, "sources_count": len(docs)}

    except Exception as exc:
        logger.warning(
            "Feedback context retrieval failed for client=%s: %s", client_id, exc
        )
        return {"injection_context": "", "sources_count": 0}