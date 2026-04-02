import asyncio
import httpx
import json

async def main():
    print("🚀 Inizio test di simulazione del Benchmark Service\n")
    
    # 1. Definiamo i dati del contratto
    fake_sector = "Legal Services"
    client_id = "test_client_benchmark_01"
    
    # Simuleremo le extraction che provengono da DSPy (con Risk Flag)
    extraction_payload = {
        "client_name": "Acme Corp Benchmark Test",
        "contract_type": "MSA",
        "total_value_eur": 850000.0,
        "risk_flags": [
            {"flag": "Limitation of Liability", "severity": "high", "clause": "L'azienda declina ogni e qualsiasi responsibilita..."}
        ]
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Ingest del contratto per accumulare storico benchmark.
        # N.B. fallirà se il db_opt_in non è TRUE, ma per questo test facciamo finta che sia la chiamata 
        # asincrona interna a funzionare. Facciamo una vera e propria GET sulla prediction del Benchmark
        
        url_summary = "http://localhost:8006/benchmark/summary"
        url_clause = "http://localhost:8006/benchmark/clause"
        
        print(f"📡 Test API 1: Riepilogo Settoriale (Quello che vede Qwen/Gemini prima di dare lo score)")
        resp = await client.get(url_summary, params={"sector": fake_sector, "deal_size": 850000.0})
        print(f"   Status: {resp.status_code}")
        print(f"   Risposta: {resp.json()}\n")

        print(f"📡 Test API 2: Analisi Clausola Specifica (Clausola Gemella per 'Limitation of Liability')")
        resp = await client.get(url_clause, params={"clause_type": "Limitation of Liability", "sector": fake_sector})
        print(f"   Status: {resp.status_code}")
        print(f"   Risposta: {resp.json()}\n")
        
        print(f"🎯 Conclusione:\nSe i valori sono stati emessi, il backend risponderà con un percentile di {resp.json().get('percentile_rank', 'N/A')}.")

if __name__ == "__main__":
    asyncio.run(main())
