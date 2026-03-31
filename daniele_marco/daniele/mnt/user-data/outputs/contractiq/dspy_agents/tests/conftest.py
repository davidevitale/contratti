"""
Shared fixtures for dspy_agents tests.
All external dependencies (Redis, LLM, RAG) are mocked —
tests run locally without Docker.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Ensure the dspy_agents package root is importable ──────────────────────
AGENTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AGENTS_ROOT not in sys.path:
    sys.path.insert(0, AGENTS_ROOT)

# ── Env vars BEFORE any module import (dspy.OpenAI reads them at import) ───
os.environ.setdefault("EXTERNAL_LLM_URL", "https://fake-regolo.example.com/v1/")
os.environ.setdefault("EXTERNAL_LLM_MODEL", "qwen-test")
os.environ.setdefault("EXTERNAL_LLM_API_KEY", "test-key")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("RAG_SERVICE_URL", "http://localhost:8002")

# ── Patch dspy.OpenAI and dspy.settings.configure at import time ───────────
_mock_lm = MagicMock(name="MockLM")

with patch("dspy.OpenAI", return_value=_mock_lm), \
     patch("dspy.settings.configure"), \
     patch("dspy.configure", create=True):
    from main import app

# ── httpx async test client ────────────────────────────────────────────────
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_redis():
    """In-memory mock that behaves like redis.asyncio."""
    mock = AsyncMock()
    mock.lrange = AsyncMock(return_value=[])
    mock.rpush = AsyncMock()
    mock.llen = AsyncMock(return_value=1)
    return mock


@pytest.fixture
def _patch_redis(mock_redis):
    """Inject mock_redis into the running app."""
    import main as m
    original = m.redis_client
    m.redis_client = mock_redis
    yield mock_redis
    m.redis_client = original


@pytest.fixture
def client():
    """Async httpx client wired to the FastAPI app (no network)."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ── Sample contract text (realistic ~600 words) ───────────────────────────
SAMPLE_CONTRACT = """
CONTRATTO DI LICENZA SOFTWARE N. 2024-0451

Tra:
  Fornitore: TechSolutions S.r.l., P.IVA 12345678901, Milano
  Cliente: Acme Manufacturing S.p.A., P.IVA 98765432109, Torino

Articolo 1 — Oggetto
Il Fornitore concede al Cliente una licenza non esclusiva per l'utilizzo della
piattaforma "FactoryOS" (il "Software"), comprensiva dei moduli:
  - FactoryOS Core: gestione produzione (€45.000/anno)
  - FactoryOS Analytics: dashboard e BI (€15.000/anno)
  - FactoryOS IoT Bridge: integrazione sensori (€10.000/anno)

Articolo 2 — Durata
Il presente contratto ha efficacia dal 01/03/2024 al 28/02/2027 (36 mesi).
Alla scadenza il contratto si rinnoverà automaticamente per periodi successivi
di 12 mesi, salvo disdetta comunicata con almeno 90 giorni di preavviso.

Articolo 3 — Corrispettivo
Il valore complessivo del contratto è pari a €210.000 (€70.000/anno x 3 anni).
Il canone annuale ricorrente (ARR) è di €70.000.
Sconto applicato: 8% sul listino standard.
Pagamento: entro 60 giorni dalla data di fatturazione.

Articolo 4 — SLA
Il Fornitore garantisce una disponibilità del Software pari al 99,5% su base
mensile (uptime). In caso di mancato rispetto dell'SLA, il Cliente ha diritto
a un credito pari al 5% del canone mensile per ogni punto percentuale di
indisponibilità eccedente.

Articolo 5 — Penali
5.1 Ritardo nella consegna: penale di €500 per ogni giorno di ritardo, fino
    a un massimo del 10% del valore annuale del contratto.
5.2 Violazione di riservatezza: penale di €50.000 per ogni violazione accertata.

Articolo 6 — Clausole di rischio
6.1 Limitazione di responsabilità: la responsabilità complessiva del Fornitore
    è limitata al 150% del valore annuale del contratto.
6.2 Clausola di esclusiva: il Cliente non potrà utilizzare software concorrente
    per le medesime funzionalità per tutta la durata del contratto.
6.3 Giurisdizione: Tribunale di Milano.
"""


@pytest.fixture
def sample_contract():
    return SAMPLE_CONTRACT
