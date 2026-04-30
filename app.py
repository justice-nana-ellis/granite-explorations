import os
import io
import base64
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional
import pandas as pd

load_dotenv()

import logging
# Suppress uvicorn's verbose INFO lines — only show warnings/errors
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# --- Models ---

MODEL_ID        = os.getenv("MODEL_ID",        "openai/gpt-oss-20b")
VISION_MODEL_ID = os.getenv("VISION_MODEL_ID", "granite3.2-vision:latest")

HF_TOKEN = os.getenv("HF_TOKEN")  # required only for HuggingFace models
VISION_API_BASE_URL = os.getenv("VISION_API_BASE_URL", "").strip()
VISION_API_KEY = os.getenv("VISION_API_KEY", "").strip()


def _is_ollama(model_id: str) -> bool:
    """Ollama model IDs use 'name:tag' format. HuggingFace uses 'org/repo'."""
    return "/" not in model_id


def _missing_client_error(model_id: str, *, vision: bool = False) -> Optional[str]:
    if not model_id.strip():
        return None
    if vision and VISION_API_BASE_URL and not VISION_API_KEY:
        return "VISION_API_KEY is required when VISION_API_BASE_URL is set."
    if vision and VISION_API_BASE_URL:
        return None
    if _is_ollama(model_id):
        return None
    if not HF_TOKEN:
        return (
            f"HF_TOKEN is required for HuggingFace model '{model_id}'. "
            "Set it in your Railway Variables or .env file."
        )
    return None


def _make_client(model_id: str, *, vision: bool = False) -> Optional[OpenAI]:
    if not model_id.strip():
        return None
    if vision and VISION_API_BASE_URL:
        if _missing_client_error(model_id, vision=True):
            return None
        return OpenAI(base_url=VISION_API_BASE_URL, api_key=VISION_API_KEY)
    if _is_ollama(model_id):
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    if _missing_client_error(model_id, vision=vision):
        return None
    return OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)


client        = _make_client(MODEL_ID)
vision_client = _make_client(VISION_MODEL_ID, vision=True)

app = FastAPI(
    title="Granite Finance API",
    description="An AI powered Foundational Model powered API for financial documents, spreadsheets, charts and chat.",
    version="2.0.0",
)


@app.on_event("startup")
async def startup():
    errors = []
    for label, c, mid in [("model", client, MODEL_ID), ("vision", vision_client, VISION_MODEL_ID)]:
        if label == "vision" and not mid.strip():
            print("  vision : disabled")
            continue
        config_error = _missing_client_error(mid, vision=(label == "vision"))
        if config_error:
            errors.append(f"  ⚠  {label}: {config_error}")
            continue
        try:
            c.chat.completions.create(
                model=mid,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            print(f"  {label:6s} : {mid}  ✓")
        except Exception as e:
            errors.append(f"  ⚠  {label}: {e}")
    if errors:
        print(f"")
        for err in errors:
            print(err)
        print(f"")
    else:
        print(f"")
        print(f"  model  : {MODEL_ID}  ✓")
        print(f"  vision : {VISION_MODEL_ID}  ✓")
        print(f"  server : http://localhost:8000  🚀")
        print(f"")


@app.middleware("http")
async def add_response_time_header(request: Request, call_next):
    """Adds X-Response-Time header (ms) to every response."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time"] = f"{elapsed_ms:.0f}ms"
    return response


SYSTEM_PROMPT = """Reasoning: low
You are a helpful financial analyst assistant powered by IBM Granite.
Answer questions clearly and precisely.
When analysing financial data, always cite specific figures and flag any risks you notice.
If you don't know something, say so clearly."""

FINANCE_ANALYSIS_SYSTEM = """Reasoning: low
You are a senior financial analyst.
Analyse the provided data or question with the following rules:
- Be precise with numbers and percentages
- Always mention key risks
- Structure your output with clear sections
- Flag any missing information that would improve the analysis"""


import re as _re

def _extract_answer(msg) -> str:
    """Strip <think>...</think> blocks from reasoning model responses."""
    text = msg.content or getattr(msg, "reasoning_content", None) or ""
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
    return text


def _stream_answer(client_obj, *, vision: bool = False, **kwargs) -> str:
    """
    Stream a chat completion and discard all tokens inside <think>...</think>.
    This prevents Vercel from buffering the full reasoning chain in memory —
    only the visible answer is accumulated and returned.
    """
    model_id = kwargs.get("model", "unknown")
    config_error = _missing_client_error(model_id, vision=vision)
    if client_obj is None or config_error:
        raise RuntimeError(config_error or f"Client for model '{model_id}' is not configured.")

    inside_think = False
    buffer = ""
    answer_parts = []

    with client_obj.chat.completions.create(stream=True, **kwargs) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content or "" if chunk.choices else ""
            buffer += delta

            while buffer:
                if inside_think:
                    end = buffer.find("</think>")
                    if end == -1:
                        buffer = ""  # still inside think, discard
                        break
                    else:
                        buffer = buffer[end + len("</think>"):]
                        inside_think = False
                else:
                    start = buffer.find("<think>")
                    if start == -1:
                        answer_parts.append(buffer)
                        buffer = ""
                        break
                    else:
                        answer_parts.append(buffer[:start])
                        buffer = buffer[start + len("<think>"):]
                        inside_think = True

    return "".join(answer_parts).strip()



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
        reply = _stream_answer(client, model=MODEL_ID, messages=messages,
                              max_tokens=request.max_tokens, temperature=request.temperature)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Granite API error: {str(e)}")

    return ChatResponse(reply=reply, model=MODEL_ID, tokens_used=None)


@app.post("/analyse")
def analyse(request: ChatRequest):
    """
    Finance-specific analysis endpoint.
    Uses a stricter system prompt focused on financial analysis.
    """
    messages = [
        {"role": "system", "content": FINANCE_ANALYSIS_SYSTEM},
        {"role": "user",   "content": request.message},
    ]

    try:
        analysis = _stream_answer(client, model=MODEL_ID, messages=messages,
                                  max_tokens=request.max_tokens or 768, temperature=0.3)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Granite API error: {str(e)}")

    return {"analysis": analysis, "model": MODEL_ID}


# ──────────────────────────────────────────────
# FILE UPLOAD ENDPOINTS
# ──────────────────────────────────────────────

@app.post("/upload")
async def upload_any(
    file: Optional[UploadFile] = File(None),
    question: Optional[str] = Form(None),
    max_tokens: int = Form(1024),
):
    """
    Universal endpoint — file is optional.
    - No file + no question  → ask user to provide a question
    - No file + question     → answer the question directly (same as /chat)
    - File + question        → analyse the file with the question
    - File + no question     → analyse the file with a default question
    Automatically routes by file extension:
      .pdf / .docx            → Docling text extraction → LLM
      .xlsx / .xls / .csv     → pandas stats            → LLM
      .png / .jpg / .jpeg / .webp → base64              → Vision model
    """
    # No file and no question — nothing to work with
    if not file and not question:
        raise HTTPException(
            status_code=400,
            detail="Please provide a question, a file, or both.",
        )

    # No file — just answer the question directly
    if not file:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
        try:
            answer = _stream_answer(client, model=MODEL_ID, messages=messages,
                                    max_tokens=max_tokens, temperature=0.7)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Granite API error: {str(e)}")
        return {"type": "chat", "answer": answer, "model": MODEL_ID, "tokens_used": None}

    # File provided — use default question if none given
    question = question or "Analyse this file and highlight the key information."

    filename = (file.filename or "").lower()
    ext = os.path.splitext(filename)[1]

    IMAGE_EXTS = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    PDF_EXTS   = {".pdf", ".docx"}
    TABLE_EXTS = {".xlsx", ".xls", ".csv"}

    if ext in IMAGE_EXTS:
        # ── Vision ──────────────────────────────────────────────────────
        if not VISION_MODEL_ID.strip():
            raise HTTPException(
                status_code=503,
                detail="Vision model is disabled for this deployment. Set VISION_MODEL_ID to a supported provider-backed vision model to enable image uploads.",
            )
        mime = IMAGE_EXTS[ext]
        contents = await file.read()
        b64 = base64.b64encode(contents).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text",      "text": question},
                ],
            }
        ]
        try:
            answer = _stream_answer(vision_client, vision=True, model=VISION_MODEL_ID, messages=messages,
                                    max_tokens=max_tokens, temperature=0.3)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Vision model error: {str(e)}")
        return {
            "file": file.filename, "type": "image",
            "answer": answer, "model": VISION_MODEL_ID, "tokens_used": None,
        }

    elif ext in PDF_EXTS:
        # ── PDF / DOCX ───────────────────────────────────────────────────
        contents = await file.read()
        try:
            if ext == ".pdf":
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(contents))
                extracted_text = "\n\n".join(
                    page.extract_text() or "" for page in reader.pages
                ).strip()
            else:  # .docx
                import docx
                doc = docx.Document(io.BytesIO(contents))
                extracted_text = "\n\n".join(p.text for p in doc.paragraphs if p.text).strip()
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not parse file: {str(e)}")
        MAX_CHARS = 8000
        truncated = len(extracted_text) > MAX_CHARS
        if truncated:
            extracted_text = extracted_text[:MAX_CHARS] + "\n\n[... truncated ...]"
        messages = [
            {"role": "system", "content": FINANCE_ANALYSIS_SYSTEM},
            {"role": "user",   "content": f"Document:\n\n{extracted_text}\n\n---\nQuestion: {question}"},
        ]
        try:
            answer = _stream_answer(client, model=MODEL_ID, messages=messages,
                                    max_tokens=max_tokens, temperature=0.3)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Granite API error: {str(e)}")
        return {
            "file": file.filename, "type": "pdf",
            "answer": answer, "model": MODEL_ID, "truncated": truncated, "tokens_used": None,
        }

    elif ext in TABLE_EXTS:
        # ── Excel / CSV ─────────────────────────────────────────────────
        contents = await file.read()
        buf = io.BytesIO(contents)
        try:
            df = pd.read_csv(buf) if ext == ".csv" else pd.read_excel(buf)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse file: {str(e)}")
        try:
            stats = df.describe().to_string()
        except Exception:
            stats = "No numeric statistics available."

        # With wide files (many columns) the preview and stats explode in size.
        # Only describe the first 10 numeric columns to keep stats manageable.
        numeric_cols = df.select_dtypes(include="number").columns[:10].tolist()
        try:
            stats = df[numeric_cols].describe().to_string() if numeric_cols else "No numeric columns."
        except Exception:
            stats = "No numeric statistics available."

        preview_df = df.iloc[:10, :10]
        MAX_CONTEXT_CHARS = 3000
        data_context = (
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            f"Columns: {', '.join(df.columns.tolist())}\n\n"
            f"--- Statistical Summary (first 10 numeric columns) ---\n{stats}\n\n"
            f"--- Data Preview (first 10 rows, first 10 columns) ---\n"
            f"{preview_df.to_string(index=False)}"
        )
        if len(data_context) > MAX_CONTEXT_CHARS:
            data_context = data_context[:MAX_CONTEXT_CHARS] + "\n\n[... truncated ...]"

        messages = [
            {"role": "system", "content": FINANCE_ANALYSIS_SYSTEM},
            {"role": "user",   "content": f"Spreadsheet:\n\n{data_context}\n\n---\nQuestion: {question}\n\nKeep your answer concise (under 400 words)."},
        ]
        try:
            answer = _stream_answer(client, model=MODEL_ID, messages=messages,
                                    max_tokens=600, temperature=0.3)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Granite API error: {str(e)}")

        # Hard cap — prevent any oversized response reaching Vercel's Lambda limit
        if len(answer) > 4000:
            answer = answer[:4000] + "\n\n[... truncated for size ...]"

        return {
            "file": file.filename, "type": "spreadsheet",
            "answer": answer,
            "model": MODEL_ID, "rows": df.shape[0], "num_columns": df.shape[1],
            "tokens_used": None,
        }

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: PDF, Word (.docx), Excel (.xlsx/.xls), CSV, PNG, JPEG, WebP.",
        )

