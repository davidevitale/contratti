import asyncio
import json
import logging
import os
import sys
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
import sqlalchemy as sa
from sqlalchemy import Text, String

import dspy
from signatures.contract_signatures import ClausolaGemella
from agents.orchestrator import ClausolaGemellaAgent
from utils.json_utils import safe_parse_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXTERNAL_LLM_URL = os.getenv("EXTERNAL_LLM_URL", "http://ollama:11434/v1/")
EXTERNAL_LLM_MODEL = os.getenv("EXTERNAL_LLM_MODEL", "qwen3.5-122b")
EXTERNAL_LLM_API_KEY = os.getenv("EXTERNAL_LLM_API_KEY", "")

_api_base = EXTERNAL_LLM_URL.rstrip("/") + "/"
ollama_model = dspy.OpenAI(
    model=EXTERNAL_LLM_MODEL,
    api_key=EXTERNAL_LLM_API_KEY,
    api_base=_api_base,
    model_type="chat",
    max_tokens=2048,
    temperature=0.2,
)
dspy.settings.configure(lm=ollama_model)

DB_URL = os.getenv("POSTGRES_URL", "postgresql+asyncpg://postgres:postgres@db:5432/contractiq")
engine = create_async_engine(DB_URL)

class Base(DeclarativeBase):
    pass

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(36))
    client_id: Mapped[str] = mapped_column(String(100))
    filename: Mapped[str] = mapped_column(String(255))
    extracted_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


async def run_batch():
    agent = ClausolaGemellaAgent()
    logger.info("Avvio Batch Clausola Gemella Portfolio...")

    async with AsyncSession(engine) as db:
        result = await db.execute(sa.select(Document).where(Document.extracted_json.isnot(None)))
        docs = result.scalars().all()
        
        updated_count = 0
        for doc in docs:
            doc_data = safe_parse_json(doc.extracted_json)
            if not isinstance(doc_data, dict):
                continue

            current_gemella = doc_data.get("clausola_gemella", [])
            # Skip if already processed
            if current_gemella and isinstance(current_gemella, list) and len(current_gemella) > 0:
                continue

            extraction = doc_data.get("extraction", {})
            risk_flags = extraction.get("risk_flags", [])
            if not risk_flags or not isinstance(risk_flags, list):
                continue
                
            logger.info(f"Elaborazione clausola gemella per documento {doc.id} ({doc.client_id})")
            
            gemella_results = []
            for flag in risk_flags[:3]:
                if not isinstance(flag, dict):
                    continue
                try:
                    clause_text = flag.get("clause", "")
                    clause_type = flag.get("flag", "general")
                    if not clause_text:
                        continue
                        
                    prediction = agent.forward(
                        clause_text=clause_text,
                        clause_type=clause_type,
                        sector="Technology",  # Si può parametrizzare se necessario
                        benchmark_corpus="",
                    )
                    
                    try:
                        # Extract JSON logic simplified since we want direct field access if possible
                        gemella_results.append({
                            "clause_type": clause_type,
                            "percentile_ranking": int(getattr(prediction, "percentile_ranking", 50)),
                            "benchmark_summary": str(getattr(prediction, "benchmark_summary", "")),
                            "deviation_analysis": str(getattr(prediction, "deviation_analysis", "")),
                            "negotiation_script": str(getattr(prediction, "negotiation_script", "")),
                            "priority": str(getattr(prediction, "priority", "medium")),
                            "benchmark_reliability": str(getattr(prediction, "benchmark_reliability", "medium")),
                        })
                    except Exception as e:
                        logger.warning(f"Failed to parse gemella output for {doc.id}: {e}")
                except Exception as e:
                    logger.error(f"Agent failed for {doc.id}: {e}")
            
            if gemella_results:
                doc_data["clausola_gemella"] = gemella_results
                doc.extracted_json = json.dumps(doc_data, ensure_ascii=False)
                updated_count += 1
                logger.info(f"Aggiunti {len(gemella_results)} gemella al documento {doc.id}")

            # Optional sleep to not overwhelm LLM
            await asyncio.sleep(1)
        
        if updated_count > 0:
            await db.commit()
            logger.info(f"Batch completato: {updated_count} documenti aggiornati.")
        else:
            logger.info("Nessun documento richiedeva aggiornamento Gemella.")

if __name__ == "__main__":
    asyncio.run(run_batch())
