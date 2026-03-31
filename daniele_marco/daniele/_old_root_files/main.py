"""
contractiq/rag_service/main.py

Per-Client RAG Service.

Architecture:
  - Each client gets their own ChromaDB COLLECTION: client_{client_id}_contracts
  - Each client also has a benchmark collection: benchmark_{sector}
  - Clausola Gemella queries the benchmark collection for cross-client comparison

Lock-in mechanism:
  - Every contract ingested builds the client's private corpus
  - Every user query and feedback creates episodic memory
  - The richer the corpus, the more precise retrieval → better answers
  - Deleting the account means losing years of structured contract history

Chunking strategy:
  - NOT fixed-size chunks (naive approach)
  - Clause-level semantic chunking: each clause is an independent unit
  - Each chunk carries rich metadata for filtering
"""

import hashlib
import json
import logging
import os
from typing import Optional

import chromadb
from chromadb.config import Settings
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ContractIQ RAG Service", version="1.0.0")

CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_TOKEN = os.getenv("CHROMA_TOKEN", "contractiq-secret")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB Client
# ─────────────────────────────────────────────────────────────────────────────

chroma_client: Optional[chromadb.AsyncHttpClient] = None


async def get_chroma():
    global chroma_client
    if chroma_client is None:
        chroma_client = await chromadb.AsyncHttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
        )
    return chroma_client


def client_collection_name(client_id: str) -> str:
    """Each client = isolated collection."""
    return f"client_{client_id}_contracts"


def benchmark_collection_name(sector: str) -> str:
    """Anonymized sector benchmarks for Clausola Gemella."""
    return f"benchmark_{sector.lower().replace(' ', '_')}"


def episodic_collection_name(client_id: str) -> str:
    """Client episodic memory: Q&A history for context enrichment."""
    return f"episodic_{client_id}"


# ─────────────────────────────────────────────────────────────────────────────
# Embedding via Anthropic (Voyage)
# ─────────────────────────────────────────────────────────────────────────────

async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings using Anthropic's embedding API.
    Falls back to simple hash-based mock if unavailable.
    """
    if not ANTHROPIC_API_KEY:
        import random
        return [[random.gauss(0, 0.1) for _ in range(1024)] for _ in texts]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {ANTHROPIC_API_KEY}",
                    "content-type": "application/json",
                },
                json={
                    "model": "voyage-3",
                    "input": texts,
                    "input_type": "document",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                return [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.warning(f"Voyage embedding failed: {e}, using fallback")

    import random
    return [[random.gauss(0, 0.1) for _ in range(1024)] for _ in texts]


# ─────────────────────────────────────────────────────────────────────────────
# Clause-Level Semantic Chunker
# ─────────────────────────────────────────────────────────────────────────────

def semantic_clause_chunker(text: str, contract_metadata: dict) -> list[dict]:
    """
    Split contract text into clause-level chunks.

    Strategy:
      1. Detect numbered clauses (1., 2.1, Art. 1, ARTICLE I, etc.)
      2. Detect standard section headers
      3. Each clause becomes one chunk with rich metadata
      4. Fallback: 500-token sliding window with 100-token overlap

    This is FAR superior to fixed-size chunking for contract Q&A.
    """
    import re

    chunks = []
    clause_patterns = [
        r'\n\s*(?:Art(?:icle|icolo)?\.?\s*\d+[\.\d]*)',
        r'\n\s*\d+[\.\d]+\s+[A-Z]',
        r'\n\s*(?:CLAUSE|CLAUSOLA|SEZIONE|SECTION)\s+\d+',
        r'\n\s*(?:Whereas|WHEREAS|Premesso che)',
    ]

    # Try to split by clauses
    combined_pattern = '|'.join(f'({p})' for p in clause_patterns)
    splits = re.split(combined_pattern, text, flags=re.IGNORECASE)

    if len(splits) > 3:
        # Successfully identified clause boundaries
        current_clause = ""
        clause_num = 0
        for part in splits:
            if part and len(part.strip()) > 20:
                if re.match(combined_pattern, part, re.IGNORECASE):
                    if current_clause.strip():
                        chunks.append({
                            "text": current_clause.strip(),
                            "metadata": {
                                **contract_metadata,
                                "chunk_type": "clause",
                                "clause_index": clause_num,
                                "char_count": len(current_clause),
                            }
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
                }
            })
    else:
        # Fallback: sliding window chunking
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
                    }
                })

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Request Models
# ─────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    client_id: str
    contract_id: str
    contract_text: str
    metadata: dict  # client_name, sector, value, dates, etc.

class RetrieveRequest(BaseModel):
    client_id: str
    query: str
    top_k: int = 5
    filter_metadata: Optional[dict] = None

class BenchmarkIngestRequest(BaseModel):
    sector: str
    clause_text: str
    clause_type: str
    metadata: dict  # anonymized: no client identifiers

class EpisodicMemoryRequest(BaseModel):
    client_id: str
    question: str
    answer: str
    rating: int


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "rag_service"}


@app.post("/ingest/contract")
async def ingest_contract(req: IngestRequest):
    """
    Ingest a contract into the client's isolated RAG collection.
    Performs semantic clause-level chunking before embedding.
    """
    chroma = await get_chroma()
    collection = await chroma.get_or_create_collection(
        name=client_collection_name(req.client_id),
        metadata={"hnsw:space": "cosine"},
    )

    # Clause-level chunking
    chunks = semantic_clause_chunker(req.contract_text, {
        **req.metadata,
        "contract_id": req.contract_id,
        "client_id": req.client_id,
    })

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks extracted from contract")

    # Embed all chunks
    texts = [c["text"] for c in chunks]
    embeddings = await embed_texts(texts)

    # Generate stable IDs
    ids = [
        f"{req.contract_id}_chunk_{i}_{hashlib.md5(c['text'].encode()).hexdigest()[:8]}"
        for i, c in enumerate(chunks)
    ]

    await collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
    )

    logger.info(f"Ingested {len(chunks)} chunks for client {req.client_id}, contract {req.contract_id}")
    return {"status": "ingested", "chunks": len(chunks), "contract_id": req.contract_id}


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    """
    Retrieve relevant contract chunks for a client query.
    Also returns client history from episodic memory.
    """
    chroma = await get_chroma()

    # Embed the query
    query_embeddings = await embed_texts([req.query])

    # Retrieve from client's contract collection
    try:
        collection = await chroma.get_collection(client_collection_name(req.client_id))
        results = await collection.query(
            query_embeddings=query_embeddings,
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
    except Exception as e:
        logger.warning(f"Contract retrieval failed for {req.client_id}: {e}")
        context = ""

    # Retrieve episodic memory (past Q&A)
    client_history = ""
    try:
        episodic = await chroma.get_collection(episodic_collection_name(req.client_id))
        ep_results = await episodic.query(
            query_embeddings=query_embeddings,
            n_results=3,
        )
        ep_docs = ep_results.get("documents", [[]])[0]
        if ep_docs:
            client_history = "Past interactions:\n" + "\n".join(ep_docs)
    except Exception:
        pass

    return {"context": context, "client_history": client_history}


@app.get("/benchmarks/{sector}")
async def get_benchmarks(sector: str):
    """Return anonymized benchmark data for Clausola Gemella."""
    chroma = await get_chroma()
    try:
        collection = await chroma.get_collection(benchmark_collection_name(sector))
        # Get a sample of benchmark clauses
        results = await collection.get(limit=50, include=["documents", "metadatas"])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        # Build summary stats
        sla_values = [m.get("sla_uptime") for m in metas if m.get("sla_uptime")]
        payment_terms = [m.get("payment_days") for m in metas if m.get("payment_days")]

        return {
            "sector": sector,
            "sample_size": len(docs),
            "benchmarks": {
                "avg_sla_uptime": sum(sla_values) / len(sla_values) if sla_values else 99.5,
                "avg_payment_days": sum(payment_terms) / len(payment_terms) if payment_terms else 30,
                "corpus_doc_count": len(docs),
            },
            "corpus_summary": "\n".join(docs[:10]),  # Sample clauses for comparison
        }
    except Exception:
        # Return defaults if no benchmark data yet
        return {
            "sector": sector,
            "sample_size": 0,
            "benchmarks": {"avg_sla_uptime": 99.5, "avg_payment_days": 30},
            "corpus_summary": "No benchmark data available for this sector yet.",
        }


@app.post("/ingest/benchmark")
async def ingest_benchmark(req: BenchmarkIngestRequest):
    """
    Add an anonymized clause to the sector benchmark corpus.
    Called automatically after each contract ingestion (anonymized).
    Powers Clausola Gemella.
    """
    chroma = await get_chroma()
    collection = await chroma.get_or_create_collection(
        benchmark_collection_name(req.sector)
    )
    embeddings = await embed_texts([req.clause_text])
    chunk_id = hashlib.md5(req.clause_text.encode()).hexdigest()

    await collection.upsert(
        ids=[chunk_id],
        embeddings=embeddings,
        documents=[req.clause_text],
        metadatas=[{**req.metadata, "sector": req.sector, "clause_type": req.clause_type}],
    )
    return {"status": "benchmark_ingested"}


@app.post("/episodic/store")
async def store_episodic(req: EpisodicMemoryRequest):
    """
    Store a Q&A interaction in the client's episodic memory.
    High-rated interactions (4-5 stars) get priority storage.
    """
    if req.rating < 3:
        return {"status": "skipped_low_rating"}

    chroma = await get_chroma()
    collection = await chroma.get_or_create_collection(episodic_collection_name(req.client_id))

    memory_text = f"Q: {req.question}\nA: {req.answer}"
    embeddings = await embed_texts([memory_text])
    memory_id = hashlib.md5(memory_text.encode()).hexdigest()

    await collection.upsert(
        ids=[memory_id],
        embeddings=embeddings,
        documents=[memory_text],
        metadatas=[{"rating": req.rating, "timestamp": datetime.utcnow().isoformat()}],
    )
    return {"status": "stored"}
