# Granite Finance API

A locally-running AI API for financial documents, spreadsheets, charts and chat — powered by IBM Granite (and any other Ollama model) via FastAPI.

---

## What it does

| Endpoint | What you send | What you get back |
|---|---|---|
| `POST /chat` | A question in JSON | AI answer |
| `POST /analyse` | Financial data or question | Structured analysis with risks |
| `POST /upload` | Any file + question | AI answer auto-routed by file type |
| `POST /upload/pdf` | PDF file + question | Answer from extracted text |
| `POST /upload/excel` | Excel / CSV + question | Answer from computed statistics |
| `POST /upload/image` | Chart / image + question | Answer from vision model |
| `GET /health` | — | `{"status": "healthy"}` |

---

## Request Flow

```
Your HTTP Request
        │
        ▼
  FastAPI receives it
        │
        ▼
  Middleware: start timer ──────────────────────────────────────┐
        │                                                        │
        ▼                                                        │
  Route handler:                                                 │
        │                                                        │
        ├─ POST /chat ─────────────────────────────────────┐    │
        │   Build messages with SYSTEM_PROMPT              │    │
        │   Send to Ollama (MODEL_ID)                      │    │
        │   Return { reply, model, tokens_used }           │    │
        │                                                  │    │
        ├─ POST /analyse ──────────────────────────────────┤    │
        │   Build messages with FINANCE_ANALYSIS_SYSTEM    │    │
        │   Send to Ollama (MODEL_ID) at temp=0.3          │    │
        │   Return { analysis, model }                     │    │
        │                                                  │    │
        └─ POST /upload ───────────────────────────────────┤    │
            Detect file extension                          │    │
                │                                          │    │
                ├─ .png/.jpg/.jpeg/.webp                   │    │
                │   Encode image as base64                 │    │
                │   Send to VISION_MODEL_ID                │    │
                │   Return { answer, model, type:"image" } │    │
                │                                          │    │
                ├─ .pdf / .docx                            │    │
                │   Save to temp file                      │    │
                │   Docling extracts text + tables         │    │
                │   Truncate to 8,000 chars if needed      │    │
                │   Send to MODEL_ID                       │    │
                │   Delete temp file                       │    │
                │   Return { answer, model, type:"pdf" }   │    │
                │                                          │    │
                └─ .xlsx / .xls / .csv                     │    │
                    pandas reads file into DataFrame        │    │
                    Compute .describe() statistics          │    │
                    Take first 30 rows as preview           │    │
                    Send context + question to MODEL_ID     │    │
                    Return { answer, model, type:           │    │
                             "spreadsheet", columns }       │    │
                                                            │    │
        ◄───────────────────────────────────────────────────┘    │
        │                                                         │
        ▼                                                         │
  Middleware: stop timer ◄─────────────────────────────────────────
  Add X-Response-Time header (ms)
        │
        ▼
  Response back to you
```

---

## Prerequisites

Install these before anything else:

1. **Python 3.10+** — [python.org](https://www.python.org/downloads/)
2. **Ollama** — [ollama.com](https://ollama.com/download)

---

## Setup (after cloning)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and set your models:

```ini
# Optional — only needed if using Hugging Face APIs
HF_TOKEN=hf_your_token_here

# Model for chat / analyse / document endpoints
# Fast:    MODEL_ID=granite3.2:2b
# Quality: MODEL_ID=granite3.3:latest
# Other:   MODEL_ID=qwen3:8b   or   MODEL_ID=gpt-oss:20b
MODEL_ID=granite3.2:2b

# Model for image / chart endpoints
VISION_MODEL_ID=granite3.2-vision:latest
```

### 5. Pull the models in Ollama

```bash
# Text model (pick one)
ollama pull granite3.2:2b        # fastest
ollama pull granite3.3:latest    # best quality
ollama pull qwen3:8b             # most accurate overall

# Vision model (for image/chart uploads)
ollama pull granite3.2-vision:latest
```

### 6. Start Ollama

On macOS — if you installed the Ollama app it auto-starts. Check with:
```bash
ollama ps
```

If nothing is running:
```bash
ollama serve
```

### 7. Start the API server

```bash
uvicorn app:app --reload
```

You should see:
```
  model  : granite3.2:2b  ✓
  vision : granite3.2-vision:latest  ✓
  server : http://localhost:8000  🚀
```

---

## Usage

Open `api.http` in VS Code with the [REST Client](https://marketplace.visualstudio.com/items?itemName=humao.rest-client) extension and click **Send Request** on any example.

Or use curl:

```bash
# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a P/E ratio?"}'

# Upload any file
curl -X POST http://localhost:8000/upload \
  -F "file=@./report.pdf" \
  -F "question=What are the key financial highlights?"
```

Interactive API docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Swapping models

Edit `.env` and restart the server — no code changes needed:

```ini
MODEL_ID=granite3.3:latest
```

Any model in your `ollama list` works.

---

## Supported file types for `/upload`

| Extension | Handler | Model used |
|---|---|---|
| `.pdf` | IBM Docling (text + table extraction) | `MODEL_ID` |
| `.docx` | IBM Docling | `MODEL_ID` |
| `.xlsx` / `.xls` | pandas statistics | `MODEL_ID` |
| `.csv` | pandas statistics | `MODEL_ID` |
| `.png` / `.jpg` / `.jpeg` / `.webp` | base64 vision | `VISION_MODEL_ID` |
