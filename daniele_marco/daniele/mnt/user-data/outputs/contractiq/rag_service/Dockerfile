# Usiamo bookworm (Debian 12) per scaricare i wheel binari di chromadb
# ed evitare la compilazione C++ di hnswlib (risparmio: 10-15 minuti)
FROM python:3.11-bookworm

WORKDIR /app

# Solo curl, niente build-essential: i wheel precompilati non richiedono compilatori
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- LAYER 1: torch CPU-only (~200 MB invece di ~2.5 GB con CUDA) ---
# Installiamo torch PRIMA di requirements.txt così pip lo riusa senza riscaricarlo
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu

# --- LAYER 2: sentence-transformers + transformers pinnato ---
# transformers>=4.47 introduce auto_docstring.py incompatibile con sentence-transformers==2.3.1
RUN pip install --no-cache-dir \
    "transformers>=4.6.0,<4.47.0" \
    sentence-transformers==2.3.1

# --- LAYER 3: resto delle dipendenze (fastapi, chromadb, ecc.) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- LAYER 4: pre-download del modello embedding (cached in image) ---
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# --- LAYER 5: codice applicativo (cambia spesso, qui per sfruttare la cache) ---
COPY . .

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
