"""
contractiq/gateway/main.py

API Gateway — single entry point for all ContractIQ services.
Handles auth, routing, rate limiting, and response caching.
Reuses the existing project/document/message structure from the team's Flask architecture.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Header

try:
    import redis.asyncio as redis
except ImportError:
    import redis as redis
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app_: FastAPI):
    """Application lifespan: connects Redis and creates DB tables."""
    global redis_client
    redis_client = redis.from_url(
        os.getenv("REDIS_URL", "redis://redis:6379/0"), decode_responses=True
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Gateway ready")
    yield

app = FastAPI(title="ContractIQ Gateway", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs
DSPY_URL = os.getenv("DSPY_AGENTS_URL", "http://dspy_agents:8001")
RAG_URL = os.getenv("RAG_SERVICE_URL", "http://rag_service:8002")
PARSER_URL = os.getenv("PARSER_SERVICE_URL", "http://parser_service:8003")
ANALYTICS_URL = os.getenv("ANALYTICS_SERVICE_URL", "http://analytics_service:8004")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploads")

# Timeout solo per parse (CPU-bound). LLM calls usano None (illimitato).
PARSE_TIMEOUT = int(os.getenv("PARSE_TIMEOUT", "180"))
CHAT_TIMEOUT = None    # Nessun timeout: l'LLM cloud può richiedere minuti

os.makedirs(UPLOAD_DIR, exist_ok=True)

engine = create_async_engine(
    os.getenv("POSTGRES_URL", "").replace("postgresql://", "postgresql+asyncpg://")
)
redis_client = None


# ─────────────────────────────────────────────────────────────────────────────
# DB Models (mirrors the team's existing models.py — extended)
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id: Mapped[str] = mapped_column(sa.String(100), index=True)
    name: Mapped[str] = mapped_column(sa.String(255))
    sector: Mapped[str] = mapped_column(sa.String(100), default="general")
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    optimization_version: Mapped[int] = mapped_column(default=0)  # Tracks DSPy optimization version


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(sa.ForeignKey("projects.id"), index=True)
    client_id: Mapped[str] = mapped_column(sa.String(100), index=True)
    filename: Mapped[str] = mapped_column(sa.String(255))
    raw_text: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    extracted_json: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    contract_score: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    model_used: Mapped[str] = mapped_column(sa.String(50), default="llama3.2:3b")
    cost_usd: Mapped[float] = mapped_column(default=0.0)
    pages: Mapped[int] = mapped_column(default=0)
    rag_ingested: Mapped[bool] = mapped_column(default=False)
    uploaded_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(sa.ForeignKey("projects.id"), index=True)
    client_id: Mapped[str] = mapped_column(sa.String(100), index=True)
    session_id: Mapped[str] = mapped_column(sa.String(100))
    role: Mapped[str] = mapped_column(sa.String(20))
    content: Mapped[str] = mapped_column(sa.Text)
    sources: Mapped[str] = mapped_column(sa.Text, default="[]")
    user_rating: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    user_comment: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class UsageStats(Base):
    __tablename__ = "usage_stats"
    id: Mapped[int] = mapped_column(primary_key=True)
    client_id: Mapped[str] = mapped_column(sa.String(100), unique=True)
    total_cost_usd: Mapped[float] = mapped_column(default=0.0)
    total_time_saved_minutes: Mapped[int] = mapped_column(default=0)
    requests_count: Mapped[int] = mapped_column(default=0)
    feedback_count: Mapped[int] = mapped_column(default=0)
    optimization_version: Mapped[int] = mapped_column(default=0)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def proxy_post(url: str, json_data: dict, timeout: float | None = None) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=json_data)
        resp.raise_for_status()
        return resp.json()


async def proxy_get(url: str, timeout: float | None = None) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


def get_client_id(x_client_id: Optional[str] = None) -> str:
    """For MVP: client_id from header. In prod: JWT-based auth."""
    return x_client_id or "default_client"


async def ensure_project_exists(
    db: AsyncSession,
    project_id: str,
    client_id: str,
    sector: str = "general",
) -> Project:
    """Get-or-create: returns the Project, creating it if missing.
    Must be called inside an active AsyncSession (caller commits)."""
    result = await db.execute(sa.select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        project = Project(
            id=project_id,
            client_id=client_id,
            name=f"Auto-created project {project_id}",
            sector=sector,
        )
        db.add(project)
        await db.flush()  # assigns PK, makes FK valid within this txn
        logger.info(f"Auto-created project {project_id} for client {client_id}")
    return project


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

# Startup gestito dal lifespan context manager (vedi sopra).


@app.get("/health")
async def health():
    return {"status": "ok", "service": "gateway"}


@app.get("/")
async def root():
    return {
        "service": "ContractIQ Gateway",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "projects": "/api/projects/",
            "documents": "/api/documents/upload/",
            "chat": "/api/chat/",
            "analytics": "/api/analytics/portfolio/",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Projects
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/projects/")
async def create_project(
    name: str = Form(...),
    sector: str = Form("general"),
    x_client_id: Optional[str] = Header(None),
):
    client_id = get_client_id(x_client_id)
    async with AsyncSession(engine) as db:
        project = Project(client_id=client_id, name=name, sector=sector)
        db.add(project)
        await db.flush()
        project_id = project.id
        await db.commit()
    return {"id": project_id, "name": name, "sector": sector, "client_id": client_id}


@app.get("/api/projects/")
async def list_projects(x_client_id: Optional[str] = Header(None)):
    client_id = get_client_id(x_client_id)
    async with AsyncSession(engine) as db:
        result = await db.execute(
            sa.select(Project)
            .where(Project.client_id == client_id)
            .order_by(Project.created_at.desc())
        )
        projects = result.scalars().all()
    return [{"id": p.id, "name": p.name, "sector": p.sector, "created_at": p.created_at.isoformat()} for p in projects]


# ─────────────────────────────────────────────────────────────────────────────
# Documents — Upload + Parse + Analyze + Ingest to RAG
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/documents/upload/")
async def upload_document(
    file: UploadFile = File(...),
    project_id: str = Form(...),
    x_client_id: Optional[str] = Header(None),
):
    """
    Full pipeline: upload → parse → extract → score → ingest RAG.
    This is the 30-second magic from the pitch.
    """
    client_id = get_client_id(x_client_id)
    doc_id = str(uuid.uuid4())

    # Save file
    file_path = f"{UPLOAD_DIR}/{doc_id}_{file.filename}"
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Step 1: Parse document → text
    parse_result = await proxy_post(
        f"{PARSER_URL}/parse",
        {"file_path": file_path, "filename": file.filename},
        timeout=PARSE_TIMEOUT,
    )
    raw_text = parse_result.get("text", "")
    pages = parse_result.get("pages", 1)

    # Step 2: Ensure project exists (get-or-create) and read sector
    # Project + Document + Stats saved in a SINGLE session to avoid FK race conditions
    async with AsyncSession(engine) as db:
        project = await ensure_project_exists(db, project_id, client_id)
        sector = project.sector
        await db.flush()

        # Step 3: DSPy analysis (extract + score + clausola gemella)
        analysis_result = await proxy_post(
            f"{DSPY_URL}/analyze/contract",
            {"client_id": client_id, "contract_text": raw_text, "sector": sector},
            timeout=None,   # LLM cloud: nessun timeout
        )
        analysis = analysis_result.get("analysis", {})
        extracted = analysis.get("extraction", {})
        scoring = analysis.get("scoring", {})
        contract_score = scoring.get("overall_score", 0)

        # Step 4: Ingest into per-client RAG
        await proxy_post(f"{RAG_URL}/ingest/contract", {
            "client_id": client_id,
            "contract_id": doc_id,
            "contract_text": raw_text,
            "metadata": {
                "client_name": extracted.get("client_name", ""),
                "contract_type": extracted.get("contract_type", ""),
                "value_eur": extracted.get("total_value_eur", 0),
                "sector": sector,
                "end_date": extracted.get("end_date", ""),
                "contract_score": contract_score,
                "filename": file.filename,
            },
        })

        # Step 5: Contribute anonymized clauses to benchmark corpus
        _rf = extracted.get("risk_flags", [])
        risk_flags = _rf if isinstance(_rf, list) else json.loads(_rf or "[]")
        for flag in risk_flags[:5]:
            if not isinstance(flag, dict):
                continue
            try:
                await proxy_post(f"{RAG_URL}/ingest/benchmark", {
                    "sector": sector,
                    "clause_text": flag.get("clause", ""),
                    "clause_type": flag.get("flag", "general"),
                    "metadata": {
                        "severity": flag.get("severity", "medium"),
                        "payment_days": extracted.get("payment_terms_days", 30),
                        "sla_uptime": extracted.get("sla_uptime_percentage", 0),
                        "discount_pct": extracted.get("discount_percentage", 0),
                    },
                })
            except Exception as e:
                logger.warning(f"Benchmark ingestion failed for flag: {e}")

        # Step 6: Save document + usage stats (same session — FK guaranteed)
        time_saved = pages * 15  # 15 min per page
        doc = Document(
            id=doc_id,
            project_id=project_id,
            client_id=client_id,
            filename=file.filename,
            raw_text=raw_text,
            extracted_json=json.dumps(analysis),
            contract_score=contract_score,
            pages=pages,
            rag_ingested=True,
            cost_usd=parse_result.get("cost_usd", 0),
        )
        db.add(doc)
        # Update usage stats
        stats_result = await db.execute(
            sa.select(UsageStats).where(UsageStats.client_id == client_id)
        )
        stats = stats_result.scalar_one_or_none()
        if stats:
            stats.total_time_saved_minutes += time_saved
            stats.requests_count += 1
        else:
            db.add(UsageStats(
                client_id=client_id,
                total_time_saved_minutes=time_saved,
                requests_count=1,
            ))
        await db.commit()

    return {
        "document_id": doc_id,
        "filename": file.filename,
        "pages": pages,
        "contract_score": contract_score,
        "extracted": extracted,
        "scoring": scoring,
        "clausola_gemella": analysis.get("clausola_gemella", []),
        "time_saved_minutes": time_saved,
    }


@app.get("/api/documents/{doc_id}/")
async def get_document(doc_id: str):
    async with AsyncSession(engine) as db:
        result = await db.execute(sa.select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": doc.id,
        "filename": doc.filename,
        "contract_score": doc.contract_score,
        "extracted": json.loads(doc.extracted_json or "{}"),
        "uploaded_at": doc.uploaded_at.isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chat — RAG Q&A with Citations
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    project_id: str
    session_id: str
    question: str


@app.post("/api/chat/")
async def chat(req: ChatRequest, x_client_id: Optional[str] = Header(None)):
    client_id = get_client_id(x_client_id)

    result = await proxy_post(
        f"{DSPY_URL}/chat/question",
        {"client_id": client_id, "session_id": req.session_id, "question": req.question},
        timeout=CHAT_TIMEOUT,
    )
    response = result.get("response", {})

    # Store message in DB (ensure project exists first to avoid FK violation)
    async with AsyncSession(engine) as db:
        await ensure_project_exists(db, req.project_id, client_id)
        # User message
        db.add(Message(
            project_id=req.project_id, client_id=client_id,
            session_id=req.session_id, role="user", content=req.question,
        ))
        # Assistant message — sources must be JSON string, not Python list
        raw_sources = response.get("sources", [])
        sources_str = json.dumps(raw_sources) if isinstance(raw_sources, list) else str(raw_sources)
        msg = Message(
            project_id=req.project_id, client_id=client_id,
            session_id=req.session_id, role="assistant",
            content=response.get("answer", ""),
            sources=sources_str,
        )
        db.add(msg)
        await db.flush()
        message_id = msg.id
        await db.commit()

    # LOCK-IN #2: Persist the user's question to the chat_memory vector collection.
    # Every insight the manager types in chat becomes searchable context for future queries.
    try:
        await proxy_post(f"{RAG_URL}/chat/memory/store", {
            "client_id": client_id,
            "session_id": req.session_id,
            "message": req.question,
        })
    except Exception as e:
        logger.warning(f"Chat memory store failed for client={client_id} session={req.session_id}: {e}")

    return {
        "message_id": message_id,
        "answer": response.get("answer", ""),
        "confidence": response.get("confidence", "medium"),
        "sources": response.get("sources", []),
        "follow_up_questions": response.get("follow_up_questions", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feedback — THE LOCK-IN MECHANISM
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    message_id: str
    rating: int  # 1-5
    comment: Optional[str] = None


@app.post("/api/feedback/")
async def submit_feedback(req: FeedbackRequest, x_client_id: Optional[str] = Header(None)):
    """
    User feedback on a response.
    Triggers DSPy per-client optimization when threshold is reached.
    This is where the moat is built — one rating at a time.
    """
    client_id = get_client_id(x_client_id)

    # Retrieve the message
    async with AsyncSession(engine) as db:
        msg_result = await db.execute(sa.select(Message).where(Message.id == req.message_id))
        msg = msg_result.scalar_one_or_none()
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found")

        # Get the preceding user question
        q_result = await db.execute(
            sa.select(Message)
            .where(
                Message.session_id == msg.session_id,
                Message.role == "user",
                Message.created_at < msg.created_at,
            )
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        user_msg = q_result.scalar_one_or_none()

        # Capture ORM fields BEFORE commit (avoids detached instance access)
        msg_session_id = msg.session_id
        msg_content = msg.content
        msg_sources = msg.sources
        user_question = user_msg.content if user_msg else ""

        # Update message with feedback
        msg.user_rating = req.rating
        msg.user_comment = req.comment
        await db.commit()

    # Use local vars from here — ORM objects are detached after session close
    if req.rating >= 4:
        try:
            await proxy_post(f"{RAG_URL}/episodic/store", {
                "client_id": client_id,
                "question": user_question,
                "answer": msg_content,
                "rating": req.rating,
                "user_comment": req.comment,
            })
        except Exception as e:
            logger.warning(f"Episodic store failed for client {client_id}: {e}")

    # Forward to DSPy service for optimization tracking
    dspy_resp = {}
    try:
        dspy_resp = await proxy_post(f"{DSPY_URL}/feedback", {
            "client_id": client_id,
            "session_id": msg_session_id,
            "question": user_question,
            "answer": msg_content,
            "sources": msg_sources,
            "user_rating": req.rating,
            "user_comment": req.comment,
        })
    except Exception as e:
        logger.warning(f"DSPy feedback forward failed for client={client_id}: {e}")

    return {
        "status": "feedback_recorded",
        "rating": req.rating,
        "optimization_status": dspy_resp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analytics — Dashboard Data
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/analytics/portfolio/")
async def portfolio_analytics(x_client_id: Optional[str] = Header(None)):
    client_id = get_client_id(x_client_id)
    return await proxy_get(f"{ANALYTICS_URL}/portfolio/{client_id}")


@app.get("/api/analytics/stats/")
async def usage_stats(x_client_id: Optional[str] = Header(None)):
    client_id = get_client_id(x_client_id)
    async with AsyncSession(engine) as db:
        result = await db.execute(sa.select(UsageStats).where(UsageStats.client_id == client_id))
        stats = result.scalar_one_or_none()

    opt_status = await proxy_get(f"{DSPY_URL}/optimization/status/{client_id}")

    return {
        "total_cost_usd": stats.total_cost_usd if stats else 0,
        "total_time_saved_minutes": stats.total_time_saved_minutes if stats else 0,
        "requests_count": stats.requests_count if stats else 0,
        "optimization": opt_status,
    }


@app.get("/api/analytics/alerts/")
async def get_alerts(x_client_id: Optional[str] = Header(None)):
    """Proactive alerts — Disruptive Move #3."""
    client_id = get_client_id(x_client_id)
    return await proxy_get(f"{ANALYTICS_URL}/alerts/{client_id}")
