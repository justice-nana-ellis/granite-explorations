# Claude Finance API

A FastAPI server that wraps the Anthropic Claude API for financial document analysis and chat. Supports streaming responses, file uploads (CSV, PDF, images, text), and uses separate models for chat speed vs. analysis quality.

## Prerequisites

- Python 3.9+
- Claude API key from [console.anthropic.com](https://console.anthropic.com/)

## Setup

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Copy `.env.example` or create a `.env` file:

```ini
CLAUDE_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-opus-4-7          # used for file analysis
CLAUDE_CHAT_MODEL=claude-haiku-4-5-20251001  # used for chat (faster)
PORT=8000
```

## Running the server

```bash
python app.py
```

Or with uvicorn directly:

```bash
uvicorn app:app --reload --port 8000
```

On startup you will see:

```
✓ Analysis model : claude-opus-4-7
✓ Chat model     : claude-haiku-4-5-20251001
✓ Server running on http://0.0.0.0:8000
```

## API Routes

### `GET /`
Health check — returns current model and port.

### `GET /health`
Returns `{"status": "healthy"}`.

### `POST /chat`
Streaming chat with Claude. Response is `text/event-stream` — tokens arrive as they are generated.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a P/E ratio?"}'
```

Optional field:
```json
{
  "message": "...",
  "system_prompt": "You are a financial advisor."
}
```

### `POST /upload`
Upload a file and ask a question about it. Response is `text/event-stream` — streams Claude's analysis as it is generated.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@report.pdf" \
  -F "question=What is the total AUM?"
```

Optional field:
```bash
-F "system_prompt=You are a portfolio analyst."
```

#### Supported file types

| Type | Handling |
|---|---|
| `.csv` | Parsed with pandas — actual totals, means, min/max per numeric column, unique date values, categorical summaries, first 5 rows |
| `.pdf` | Sent as a base64 document block |
| `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` | Sent as a base64 image block |
| `.txt`, `.html`, `.md` | Sent as inline text |

> If the client does not set a `Content-Type` header, the type is inferred from the file extension automatically.

## Models

| Env var | Default | Used for |
|---|---|---|
| `CLAUDE_MODEL` | `claude-opus-4-7` | File analysis (`/upload`) |
| `CLAUDE_CHAT_MODEL` | `claude-haiku-4-5-20251001` | Chat (`/chat`) |

Available models:
- `claude-opus-4-7` — most capable, slower
- `claude-sonnet-4-6` — balanced
- `claude-haiku-4-5-20251001` — fastest

## Interactive API docs

Run the server and visit: [http://localhost:8000/docs](http://localhost:8000/docs)
