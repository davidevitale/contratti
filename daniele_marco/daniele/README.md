# ContractIQ — Contract Intelligence OS Multi-Agente

ContractIQ è una piattaforma enterprise ad architettura a **Microservizi** progettata per automatizzare, analizzare e governare l'intero ciclo di vita dei contratti aziendali. A differenza dei classici "wrapper" su ChatGPT, ContractIQ utilizza un vero e proprio **Sistema Multi-Agente (basato sul framework DSPy)** e un motore di **Memoria Episodica (RAG)** isolato e iper-inquadrato per ogni singolo tenant aziendale.

L'architettura garantisce massima scalabilità, affidabilità e, soprattutto, impara nel tempo adattandosi alle specifiche policy di chi la utilizza.

Tutto il codice sorgente reale risiede nella directory `mnt/user-data/outputs/contractiq/`.

---

## 🎯 Value Proposition per la Vendita (Perché ContractIQ?)

Se devi presentare e vendere questa soluzione, ecco i 3 pilastri fondamentali che la differenziano drasticamente dai competitor sul mercato:

1. **Intelligenza Multi-Agente "Su Misura" e Lock-in Tecnologico 🧠**
   Non usiamo un singolo prompt generico. Il sistema è un team di molteplici *"Agenti Specializzati"* (es. l'Agente Estrazione, l'Agente Scoring, l'Agente Ricerca Anomalie). La vera magia risiede nel modulo **Optimizer**: quando un cliente interagisce col sistema e fornisce feedback positivi/negativi, l'OS riscrive dinamicamente in background (usando la telemetria di DSPy) le proprie logiche interne per "copiare" l'intuito del risk manager. Questo crea un modello proprietario quasi imbattibile e genera un formidabile **lock-in**: il cliente non vorrà più cambiare strumento perché dopo qualche mese l'AI conoscerà a memoria il *suo* approccio di lavoro specifico.

2. **Memoria Episodica e RAG Isolato per Cliente 📚**
   Ogni cliente ha il proprio Vector Database isolato (**ChromaDB**). La piattaforma non compie banali ricerche testuali: il testo contrattuale viene scomposto intelligentemente in `Celle Semantiche` a livello di clausola. Inoltre, il sistema registra una "Memoria Episodica": le risposte valutate positivamente dal management (rating >= 3) vengono permanentemente iniettate nel database semantico in forma vettoriale. Dunque l'AI, col tempo, baserà le risposte non solo sul documento aperto, ma su tutti i *casi pregressi aziendali*.

3. **Indipendenza e Privacy Assoluta dei Dati (LLM Agnostic) 🛡️**
   A differenza dei prodotti vincolati alle API di OpenAI e Anthropic, la nostra infrastruttura è agnostica. È capace di usare grandi server esterni tramite API, ma è anche progettata nativamente (tramite parametri preconfigurati) per usare **LLM Open Source Locali (es. Llama/Ollama)**, unendo la ricerca semantica gestita internamente con `all-MiniLM-L6-v2`. Questo abbatte a zero il rischio di data leak dei contratti core business o violazioni GDPR legate a server extra-europei, offrendo anche pacchetti "on-premise".

---

## 🏗️ Flusso Architetturale e Microservizi

L'infrastruttura è containerizzata con **Docker Compose** e abbraccia 6 entità isolate e ultra-modulari:

### 1. API Gateway (`gateway/` - Porta 8000)
È il traffico e il vigile di tutto il sistema.
- Tutte le chiamate applicative partono e ritornano da qui.
- Usa **PostgreSQL** per tutto ciò che è relazionale e persistente (anagrafiche utenti, documenti, log valutazioni, chat storage).
- Usa **Redis** per accodare micro-task asincroni (es. lunghi PDF) e caching dello stato, prevenendo interruzioni di servizio in caso di picchi.

### 2. DSPy Multi-Agent System (`dspy_agents/` - Porta 8001)
È il muscolo strategico dell'Intelligenza Artificiale. Governa e modella i calcoli LLM.
- **Agenti e Signatures (`contract_signatures.py`)**: Prompt in format Object-Oriented fortemente tipizzati. Troviamo ad esempio `ContractExtraction` per processare i raw-data, e `ContractScoring` per tirare fuori metriche e allerte quantitative basate sul rischio.
- **Feedback Optimizer (`optimizer.py`)**: Il motore invisibile che si attiva autonomamente una volta raggiunta la soglia prestabilita (es. `FEEDBACK_THRESHOLD=20`). Ricostruisce al volo dei programmi DSPy e salva i pesi nella directory permanente `optimized_models/`, passando da un bot standardizzato a un esperto di dominio cucito sull'utente.

### 3. RAG Service (`rag_service/` - Porta 8002)
Specialista per la ricerca semantica con Retrieval Augmented Generation rapida e non allucinata.
- Frammenta i file caricati estrapolando metadati contestuali ed evadendo le tabelle complesse.
- Trasforma le interrogazioni umane in embedding matematici paragonandoli con i paragrafi, integrandosi anche con la cronologia storica memorizzata per garantire conformità sulle Policy passate.

### 4. Parser Service (`parser_service/` - Porta 8003)
Il "colletto bianco" adibito allo sporco mestiere di parsare la documentazione.
- Ingerisce decine di formati tra cui PDF standard, scansioni in bassa risoluzione e immagini, occupandosi integralmente di svolgere l'OCR di primo livello e normalizzando il testo puro affinché gli agenti DSPy possano operare su testi puliti.

### 5. Analytics Service (`analytics_service/` - Porta 8004)
Responsabile della cruscottistica, aggregation pattern e proattività sul portafoglio clienti.
- Elabora metriche globali.
- Individui pattern nei contratti stipulati e gestisce gli **Alert**: scadenze di rinnovo in stato critico e avvisi tempestivi e sintetici per i dipartimenti legali o i responsabili commerciali.

---

## ⚙️ Workflow Tipico della Macchina (Come interagiscono gli agenti)

1. **Upload File**: Il workflow si innesca quando un Risk/Legal Manager carica un documento dal Frontend Vue/React.
2. **Normalizzazione**: Il portinaio `Gateway` manda il blob binario al `Parser Service` e attende la conversione in testo grezzo digitale esplorabile.
3. **Brainstorming Multi-Agente**: Una volta ripulito, il Gateway interpella i `dspy_agents`. L'Orchestratore ingaggia a cascata fino a 4 agenti DSPy. Il primo astrae i punti chiave. Il successivo incrocia parametri, un altro ancora valuta la severity (rischiosità). Il risultato genera il referto completo di metriche KPI (`ContractIQ Score`).
4. **Vettorizzazione in Background**: Mentre l'utente legge i risultati in real time, il `RAG Service` inietta ogni clausola identificata nel Database Vettoriale (`ChromaDB`) nel namespace univoco *di quel singolo cliente*.
5. **Chat ed Episodic Learning**: L'utente fa QA sul documento e giudica le risposte come buone/incongrue; l'input valoriale viene salvato nel Postgres e successivamente catturato dal RAG e dal modulo Optimizer per allineare l'iper-specializzazione al cliente.

---

## 🚀 Quick Start / Setup Per Demo e Deploy

```bash
# 1. Configurazione: clona il template per l'ambiente (modifica LLM e DB credentials se serve)
cp .env.example .env

# 2. Crea l'ambiente ed istanzia tutta la flotta di agenti e db (dalla porta 80 alla porta 8005)
docker compose up --build
```

### URL di Riferimento Dopo Il Boot:
- **Piattaforma Web (Frontend)**: `http://localhost:3000`
- **Main API Gateway**: `http://localhost:8000`
- **Documentazione API (Swagger/OpenAPI)**: `http://localhost:8000/docs`
