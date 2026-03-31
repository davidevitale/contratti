# ContractIQ - Quick Setup for Hackathon

## Prerequisites

1. **Docker & Docker Compose** (required)
2. **LLM API** - Choose one:
   - **Option A (Recommended)**: Run Ollama locally with `llama3.2`:
     ```bash
     curl -fsSL https://ollama.com/install.sh | sh
     ollama pull llama3.2
     ollama serve  # Runs on http://localhost:11434
     ```
   - **Option B**: Use your existing LLM API (update `.env` with your endpoint)

## Quick Start (5 minutes)

### 1. Clone & Setup
```bash
cd C:\Users\danie\Desktop\daniele

# Copy environment file (already done)
# .env is pre-configured for local development
```

### 2. Start All Services
```bash
docker compose up --build
```

This will start:
- **Frontend**: http://localhost:3000
- **API Gateway**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432
- **ChromaDB**: localhost:8005
- **Redis**: localhost:6379

### 3. Verify Everything is Running
```bash
# Check health of all services
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health

# Or visit http://localhost:3000 in your browser
```

## Troubleshooting

### Port Already in Use
If you get "port already in use" errors:
```bash
# Stop all containers
docker compose down

# Free the ports and restart
docker compose up --build
```

### LLM Not Responding
If the AI doesn't respond:
1. Check if Ollama is running: `ollama list`
2. Pull the model: `ollama pull llama3.2`
3. Or update `.env` with your LLM endpoint

### Database Issues
```bash
# Reset database
docker compose down -v
docker compose up -d postgres
# Wait 10 seconds for initialization
docker compose up -d
```

### View Logs
```bash
# All logs
docker compose logs -f

# Specific service
docker compose logs -f gateway
docker compose logs -f dspy_agents
docker compose logs -f rag_service
```

## Architecture Overview

```
┌──────────────┐
│  Frontend    │ :3000
│   (Nginx)    │
└──────┬───────┘
       │
┌──────▼─────────────────────────────┐
│  API Gateway  :8000                │
│  - Auth        │                   │
│  - Routing     │                   │
│  - Rate Limit  │                   │
└──────┬─────────┘                   │
       │                             │
   ┌───┴────┬────────┬────────┬──────┘
   │        │        │        │
┌──▼──┐ ┌──▼──┐ ┌───▼──┐ ┌───▼──┐
│DSPy │ │ RAG │ │Parser│ │Analyt│
│Agents│ │Serv │ │Serv  │ │ics   │
└──────┘ └─────┘ └──────┘ └──────┘
   │        │        │        │
┌──▼────┬───▼───┬────┴──┬─────┴──┐
│Postgres│Redis │Chroma │ Ollama │
└────────┴──────┴───────┴────────┘
```

## Key Endpoints

### Upload & Analyze Contract
```bash
curl -X POST http://localhost:8000/api/documents/upload/ \
  -F "file=@/path/to/contract.pdf" \
  -F "project_id=your-project-id" \
  -H "x-client-id: your-client-id"
```

### Chat with Contracts
```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -H "x-client-id: your-client-id" \
  -d '{
    "project_id": "your-project-id",
    "session_id": "session-123",
    "question": "What are the payment terms?"
  }'
```

### Get Dashboard Data
```bash
curl http://localhost:8000/api/analytics/portfolio/ \
  -H "x-client-id: your-client-id"
```

## For the Hackathon Presentation

1. **Start fresh**: `docker compose down -v && docker compose up --build`
2. **Wait 60 seconds** for all services to initialize
3. **Open**: http://localhost:3000
4. **Demo flow**:
   - Upload a contract PDF
   - Show the ContractIQ Score
   - Ask questions via chat
   - Show dashboard analytics
   - Demonstrate Clausola Gemella comparison

## Development Tips

- **Hot reload**: Services restart automatically on code changes
- **Debug mode**: Add `debug=true` to query params for verbose logs
- **Reset everything**: `docker compose down -v` (deletes all data)

## Need Help?

Check logs:
```bash
docker compose logs gateway dspy_agents rag_service
```

---

**Good luck with your hackathon! 🚀**
