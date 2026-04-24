import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional

load_dotenv()

# Ollama runs locally — no token needed
# granite3.2:2b is ~4x faster than granite3.3 (8B) on CPU
MODEL_ID = "granite3.2:2b"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama doesn't need a real key, but the param is required
)

app = FastAPI(
    title="Granite Finance API",
    description="A simple API wrapping IBM Granite via Hugging Face Inference API",
    version="1.0.0",
)


@app.on_event("startup")
async def warmup():
    """Send a tiny request at startup so the first real request isn't slow."""
    try:
        client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        print(f"Model '{MODEL_ID}' warmed up and ready.")
    except Exception:
        print(f"Warning: could not warm up model '{MODEL_ID}'. Is Ollama running?")


@app.middleware("http")
async def add_response_time_header(request: Request, call_next):
    """Adds X-Response-Time header (ms) to every response."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time"] = f"{elapsed_ms:.0f}ms"
    return response


SYSTEM_PROMPT = """You are a helpful financial analyst assistant powered by IBM Granite.
Answer questions clearly and precisely. 
When analysing financial data, always cite specific figures and flag any risks you notice.
If you don't know something, say so clearly."""


# --- Request / Response models ---

class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None   # override the default system prompt if needed
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7


class ChatResponse(BaseModel):
    reply: str
    model: str
    tokens_used: Optional[int] = None


class MultiTurnRequest(BaseModel):
    messages: list[dict]   # list of {"role": "user"/"assistant"/"system", "content": "..."}
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7


# --- Routes ---

@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_ID}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Single-turn chat endpoint.
    Send a message and get a reply from Granite.
    """
    system = request.system_prompt or SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": request.message},
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Granite API error: {str(e)}")

    reply = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else None

    return ChatResponse(reply=reply, model=MODEL_ID, tokens_used=tokens)


@app.post("/chat/multi-turn", response_model=ChatResponse)
def chat_multi_turn(request: MultiTurnRequest):
    """
    Multi-turn chat endpoint.
    Pass the full conversation history as a list of messages.
    Each message must have 'role' (system/user/assistant) and 'content'.

    Example body:
    {
        "messages": [
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": "What is a P/E ratio?"},
            {"role": "assistant", "content": "A P/E ratio is..."},
            {"role": "user", "content": "How does it compare to P/B ratio?"}
        ]
    }
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages list cannot be empty.")

    # Inject default system prompt if none provided
    has_system = any(m.get("role") == "system" for m in request.messages)
    messages = request.messages if has_system else [
        {"role": "system", "content": SYSTEM_PROMPT},
        *request.messages,
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Granite API error: {str(e)}")

    reply = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else None

    return ChatResponse(reply=reply, model=MODEL_ID, tokens_used=tokens)


@app.post("/analyse")
def analyse(request: ChatRequest):
    """
    Finance-specific analysis endpoint.
    Uses a stricter system prompt focused on financial analysis.
    """
    finance_system = """You are a senior financial analyst.
Analyse the provided data or question with the following rules:
- Be precise with numbers and percentages
- Always mention key risks
- Structure your output with clear sections
- Flag any missing information that would improve the analysis"""

    messages = [
        {"role": "system", "content": finance_system},
        {"role": "user",   "content": request.message},
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=request.max_tokens or 768,
            temperature=0.3,  # lower temperature for factual analysis
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Granite API error: {str(e)}")

    return {"analysis": response.choices[0].message.content, "model": MODEL_ID}
