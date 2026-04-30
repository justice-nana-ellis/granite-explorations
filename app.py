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


def _is_ollama(model_id: str) -> bool:
    """Ollama model IDs use 'name:tag' format. HuggingFace uses 'org/repo'."""
    return "/" not in model_id


def _make_client(model_id: str) -> OpenAI:
    if _is_ollama(model_id):
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    if not HF_TOKEN:
        raise RuntimeError(
            f"HF_TOKEN is required for HuggingFace model '{model_id}'. "
            "Set it in your .env file."
        )
    return OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)


client        = _make_client(MODEL_ID)
vision_client = _make_client(VISION_MODEL_ID)

app = FastAPI(
    title="Granite Finance API",
    description="An AI powered Foundational Model powered API for financial documents, spreadsheets, charts and chat.",
    version="2.0.0",
)


@app.on_event("startup")
async def startup():
    errors = []
    for label, c, mid in [("model", client, MODEL_ID), ("vision", vision_client, VISION_MODEL_ID)]:
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


SYSTEM_PROMPT = """You are a helpful financial analyst assistant powered by IBM Granite.
Answer questions clearly and precisely. 
When analysing financial data, always cite specific figures and flag any risks you notice.
If you don't know something, say so clearly."""

FINANCE_ANALYSIS_SYSTEM = """You are a senior financial analyst.
Analyse the provided data or question with the following rules:
- Be precise with numbers and percentages
- Always mention key risks
- Structure your output with clear sections
- Flag any missing information that would improve the analysis"""


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
    messages = [
        {"role": "system", "content": FINANCE_ANALYSIS_SYSTEM},
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
            response = client.chat.completions.create(
                model=MODEL_ID, messages=messages,
                max_tokens=max_tokens, temperature=0.7,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Granite API error: {str(e)}")
        return {
            "type": "chat",
            "answer": response.choices[0].message.content,
            "model": MODEL_ID,
            "tokens_used": response.usage.total_tokens if response.usage else None,
        }

    # File provided — use default question if none given
    question = question or "Analyse this file and highlight the key information."

    filename = (file.filename or "").lower()
    ext = os.path.splitext(filename)[1]

    IMAGE_EXTS = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    PDF_EXTS   = {".pdf", ".docx"}
    TABLE_EXTS = {".xlsx", ".xls", ".csv"}

    if ext in IMAGE_EXTS:
        # ── Vision ──────────────────────────────────────────────────────
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
            response = vision_client.chat.completions.create(
                model=VISION_MODEL_ID, messages=messages,
                max_tokens=max_tokens, temperature=0.3,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Vision model error: {str(e)}")
        return {
            "file": file.filename, "type": "image",
            "answer": response.choices[0].message.content,
            "model": VISION_MODEL_ID,
            "tokens_used": response.usage.total_tokens if response.usage else None,
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
            response = client.chat.completions.create(
                model=MODEL_ID, messages=messages,
                max_tokens=max_tokens, temperature=0.3,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Granite API error: {str(e)}")
        return {
            "file": file.filename, "type": "pdf",
            "answer": response.choices[0].message.content,
            "model": MODEL_ID, "truncated": truncated,
            "tokens_used": response.usage.total_tokens if response.usage else None,
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

        # With wide files (many columns) the preview explodes in size.
        # Cap preview to 10 columns and 20 rows, then hard-limit total chars.
        preview_df = df.iloc[:20, :10]
        MAX_CONTEXT_CHARS = 6000
        data_context = (
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            f"Columns: {', '.join(df.columns.tolist())}\n\n"
            f"--- Statistical Summary ---\n{stats}\n\n"
            f"--- Data Preview (first 20 rows, first 10 columns) ---\n"
            f"{preview_df.to_string(index=False)}"
        )
        if len(data_context) > MAX_CONTEXT_CHARS:
            data_context = data_context[:MAX_CONTEXT_CHARS] + "\n\n[... truncated ...]"

        messages = [
            {"role": "system", "content": FINANCE_ANALYSIS_SYSTEM},
            {"role": "user",   "content": f"Spreadsheet:\n\n{data_context}\n\n---\nQuestion: {question}"},
        ]
        try:
            response = client.chat.completions.create(
                model=MODEL_ID, messages=messages,
                max_tokens=max_tokens, temperature=0.3,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Granite API error: {str(e)}")

        msg = response.choices[0].message
        # Dump every field on the message so we can see where the answer lives
        # print("DEBUG msg dict:", msg.model_dump() if hasattr(msg, "model_dump") else vars(msg))
        answer = msg.content or getattr(msg, "reasoning_content", None) or ""

        return {
            "file": file.filename, "type": "spreadsheet",
            "answer": answer,
            "model": MODEL_ID, "rows": df.shape[0], "num_columns": df.shape[1],
            "tokens_used": response.usage.total_tokens if response.usage else None,
        }

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: PDF, Word (.docx), Excel (.xlsx/.xls), CSV, PNG, JPEG, WebP.",
        )

