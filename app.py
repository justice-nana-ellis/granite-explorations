import os
import io
import base64
import time
import tempfile
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
# Text/analysis:  granite3.2:2b     (1.6 GB, ~3-4x faster than granite3.3 — ollama pull granite3.2:2b)
# Vision/charts:  granite3.2-vision (2.4 GB, already pulled, running on GPU)
MODEL_ID        = "granite3.2:2b"
VISION_MODEL_ID = "granite3.2-vision:latest"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama doesn't need a real key, but the param is required
)

app = FastAPI(
    title="Granite Finance API",
    description="IBM Granite-powered API for financial documents, spreadsheets, charts and chat.",
    version="2.0.0",
)


@app.on_event("startup")
async def startup():
    try:
        client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        print(f"")
        print(f"  model  : {MODEL_ID}  ✓")
        print(f"  vision : {VISION_MODEL_ID}  ✓")
        print(f"  server : http://localhost:8000  🚀")
        print(f"")
    except Exception:
        print(f"")
        print(f"  ⚠  Ollama not responding — run: ollama serve")
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
    file: UploadFile = File(...),
    question: str = Form("Analyse this file and answer any key questions about the data."),
    max_tokens: int = Form(1024),
):
    """
    Universal upload endpoint — send any file and ask any question.
    Automatically routes based on extension:
      .pdf              → Docling text extraction → Granite LLM
      .xlsx / .xls / .csv → pandas stats         → Granite LLM
      .png / .jpg / .jpeg / .webp → base64        → Granite Vision
    """
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
            response = client.chat.completions.create(
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
        # ── PDF ─────────────────────────────────────────────────────────
        from docling.document_converter import DocumentConverter
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        try:
            result = DocumentConverter().convert(tmp_path)
            extracted_text = result.document.export_to_markdown()
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not parse PDF: {str(e)}")
        finally:
            os.unlink(tmp_path)
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
        data_context = (
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            f"Columns: {', '.join(df.columns.tolist())}\n\n"
            f"--- Statistical Summary ---\n{stats}\n\n"
            f"--- Data Preview (first 30 rows) ---\n{df.head(30).to_string(index=False)}"
        )
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
        return {
            "file": file.filename, "type": "spreadsheet",
            "answer": response.choices[0].message.content,
            "model": MODEL_ID, "rows": df.shape[0], "columns": df.columns.tolist(),
            "tokens_used": response.usage.total_tokens if response.usage else None,
        }

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: PDF, Word (.docx), Excel (.xlsx/.xls), CSV, PNG, JPEG, WebP.",
        )

@app.post("/upload/pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    question: str = Form("Summarise this document and highlight key financial figures."),
    max_tokens: int = Form(1024),
):
    """
    Upload a PDF (annual report, earnings release, filing).
    Docling extracts the text and tables, then Granite answers your question.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    from docling.document_converter import DocumentConverter

    contents = await file.read()
    # Docling needs a file path, so write to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        extracted_text = result.document.export_to_markdown()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse PDF: {str(e)}")
    finally:
        os.unlink(tmp_path)

    # Truncate to avoid LLM context overflow (~8 000 chars ≈ 2 000 tokens)
    MAX_CHARS = 8000
    truncated = len(extracted_text) > MAX_CHARS
    if truncated:
        extracted_text = extracted_text[:MAX_CHARS] + "\n\n[... document truncated for context limit ...]"

    messages = [
        {"role": "system", "content": FINANCE_ANALYSIS_SYSTEM},
        {"role": "user",   "content": f"Document content:\n\n{extracted_text}\n\n---\nQuestion: {question}"},
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Granite API error: {str(e)}")

    return {
        "answer": response.choices[0].message.content,
        "model": MODEL_ID,
        "chars_extracted": len(extracted_text),
        "truncated": truncated,
        "tokens_used": response.usage.total_tokens if response.usage else None,
    }


@app.post("/upload/excel")
async def upload_excel(
    file: UploadFile = File(...),
    question: str = Form("Summarise this data and identify key trends."),
    max_tokens: int = Form(768),
):
    """
    Upload an Excel (.xlsx / .xls) or CSV file.
    Pandas computes statistics, then Granite answers questions like
    'what is the sum of column X?' or 'which month had the highest revenue?'
    """
    filename = file.filename.lower()
    if not (filename.endswith(".xlsx") or filename.endswith(".xls") or filename.endswith(".csv")):
        raise HTTPException(status_code=400, detail="Only .xlsx, .xls, or .csv files are supported.")

    contents = await file.read()
    buf = io.BytesIO(contents)

    try:
        df = pd.read_csv(buf) if filename.endswith(".csv") else pd.read_excel(buf)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {str(e)}")

    # Build a rich but compact text context for the LLM
    shape_info   = f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"
    columns_info = f"Columns: {', '.join(df.columns.tolist())}"

    try:
        stats = df.describe().to_string()
    except Exception:
        stats = "No numeric statistics available."

    preview = df.head(30).to_string(index=False)

    data_context = (
        f"{shape_info}\n{columns_info}\n\n"
        f"--- Statistical Summary ---\n{stats}\n\n"
        f"--- Data Preview (first 30 rows) ---\n{preview}"
    )

    messages = [
        {"role": "system", "content": FINANCE_ANALYSIS_SYSTEM},
        {"role": "user",   "content": f"Spreadsheet data:\n\n{data_context}\n\n---\nQuestion: {question}"},
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Granite API error: {str(e)}")

    return {
        "answer": response.choices[0].message.content,
        "model": MODEL_ID,
        "rows": df.shape[0],
        "columns": df.columns.tolist(),
        "tokens_used": response.usage.total_tokens if response.usage else None,
    }


@app.post("/upload/image")
async def upload_image(
    file: UploadFile = File(...),
    question: str = Form("Describe this chart. What trends or patterns do you see? Extract any key figures."),
    max_tokens: int = Form(768),
):
    """
    Upload a chart or image (PNG, JPEG, WebP).
    Uses Granite Vision to answer questions like 'what is the trend?' or 'what is the highest value?'
    Requires: ollama pull granite3.2-vision
    """
    EXT_TO_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    # Prefer extension-derived MIME — REST clients often send wrong content_type for binary parts
    mime = EXT_TO_MIME.get(ext) or (file.content_type if file.content_type in EXT_TO_MIME.values() else None)
    if not mime:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext or file.content_type}'. Use PNG, JPEG, or WebP.",
        )

    contents = await file.read()
    b64_image = base64.b64encode(contents).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_image}"}},
                {"type": "text",      "text": question},
            ],
        }
    ]

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL_ID,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Vision model error: {str(e)}. Is '{VISION_MODEL_ID}' pulled? Run: ollama pull granite3.2-vision",
        )

    return {
        "answer": response.choices[0].message.content,
        "model": VISION_MODEL_ID,
        "tokens_used": response.usage.total_tokens if response.usage else None,
    }

