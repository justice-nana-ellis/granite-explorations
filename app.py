import asyncio
import json
import os
import base64
import io
import logging
import mimetypes
from pathlib import Path
from uuid import uuid4
from time import time
import requests
import pandas as pd
import cloudinary
import cloudinary.uploader
from anthropic import AsyncAnthropic
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncIterator
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

api_key = os.getenv("CLAUDE_API_KEY")
if not api_key:
    raise ValueError("CLAUDE_API_KEY environment variable not set")

client = AsyncAnthropic(api_key=api_key)
model = os.getenv("CLAUDE_MODEL", "claude-opus-4-7")
chat_model = os.getenv("CLAUDE_CHAT_MODEL", "claude-haiku-4-5-20251001")
port = int(os.getenv("PORT", 8000))

app = FastAPI(
    title="Claude Finance API",
    description="Simple Claude API wrapper for financial analysis",
    version="1.0.0",
)

SESSION_STORE = os.getenv("SESSION_STORE", "memory")        # only "memory" supported for now
SESSION_TTL = int(os.getenv("SESSION_TTL", 3600))           # seconds of inactivity before expiry
SESSION_MAX_MESSAGES = int(os.getenv("SESSION_MAX_MESSAGES", 10))   # rolling window size
SESSION_MAX_COUNT = int(os.getenv("SESSION_MAX_COUNT", 200))        # max concurrent sessions

# session_id -> {"messages": [...], "display": [...], "file": str|None, "system": str, "last_accessed": float}
sessions: dict[str, dict] = {}


def _delete_cloudinary_file(session: dict) -> None:
    if not session.get("cloudinary_id"):
        return
    try:
        filename = session.get("file", "")
        resource_type = "image" if filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")) else "raw"
        cloudinary.uploader.destroy(session["cloudinary_id"], resource_type=resource_type)
    except Exception:
        pass


def _purge_expired():
    now = time()
    expired = [sid for sid, s in sessions.items() if now - s["last_accessed"] > SESSION_TTL]
    for sid in expired:
        _delete_cloudinary_file(sessions[sid])
        del sessions[sid]


def _get_or_create_session(sid: str, system: str) -> dict:
    _purge_expired()
    if sid in sessions:
        sessions[sid]["last_accessed"] = time()
        return sessions[sid]
    if len(sessions) >= SESSION_MAX_COUNT:
        # evict the oldest session
        oldest = min(sessions, key=lambda k: sessions[k]["last_accessed"])
        _delete_cloudinary_file(sessions[oldest])
        del sessions[oldest]
    sessions[sid] = {
        "system": system,
        "messages": [],
        "display": [],
        "file": None,
        "cloudinary_id": None,
        "cloudinary_url": None,
        "df": None,
        "df_summary": None,      # cached once on upload, reused on every /chat
        "last_accessed": time(),
    }
    return sessions[sid]


def _reload_df_from_cloudinary(session: dict) -> None:
    if not session.get("cloudinary_url") or not session.get("file"):
        return
    try:
        resp = requests.get(session["cloudinary_url"], timeout=30)
        resp.raise_for_status()
        raw = resp.content
        filename: str = session["file"]
        if filename.endswith(".csv"):
            session["df"] = pd.read_csv(io.BytesIO(raw), low_memory=False)
        elif filename.endswith((".xlsx", ".xls")):
            session["df"] = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        if session["df"] is not None:
            session["df_summary"] = _df_summary(session["df"], filename)
    except Exception:
        pass


def _session_meta(sid: str) -> dict:
    s = sessions[sid]
    return {"session_id": sid, "file": s["file"], "messages": s["display"]}


class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    model: str


@app.on_event("startup")
async def startup():
    print(f"✓ Analysis model : {model}")
    print(f"✓ Chat model     : {chat_model}")
    print(f"✓ Server running on http://0.0.0.0:{port}")


@app.get("/")
def root():
    return {"status": "ok", "model": model, "port": port}


@app.get("/health")
def health():
    return {"status": "healthy"}


async def _stream_and_save(
    session_id: str,
    use_model: str,
    max_tokens: int,
) -> AsyncIterator[str]:
    session = sessions[session_id]
    full_reply = []

    async with client.messages.stream(
        model=use_model,
        max_tokens=max_tokens,
        system=session["system"],
        messages=session["messages"],
    ) as stream:
        async for text in stream.text_stream:
            full_reply.append(text)
            yield f"data: {text}\n\n"

    reply_text = "".join(full_reply)
    session["messages"].append({"role": "assistant", "content": reply_text})
    session["display"].append({"role": "assistant", "content": reply_text})


@app.post("/chat")
async def chat(request: ChatRequest):
    sid = request.session_id or str(uuid4())
    session = _get_or_create_session(
        sid, request.system_prompt or "You are a helpful financial assistant."
    )

    if len(session["messages"]) >= SESSION_MAX_MESSAGES:
        first = session["messages"][:1]
        rest = session["messages"][1:][-(SESSION_MAX_MESSAGES - 2):]
        session["messages"] = first + rest

    # If df was lost (e.g. server restart), reload it from Cloudinary
    if session.get("df") is None and session.get("cloudinary_url"):
        _reload_df_from_cloudinary(session)

    # Prepend the cached file summary so Claude always has the full data context
    if session.get("df_summary"):
        message_with_context = f"[File context]\n{session['df_summary']}\n\n[Question]\n{request.message}"
    else:
        message_with_context = request.message

    session["messages"].append({"role": "user", "content": message_with_context})
    session["display"].append({"role": "user", "content": request.message})

    async def _generate():
        async for chunk in _stream_and_save(sid, chat_model, 1024):
            yield chunk
        sessions[sid]["last_accessed"] = time()
        yield f"event: meta\ndata: {json.dumps(_session_meta(sid))}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


EXCEL_TYPES = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/vnd.ms-excel",                                            # .xls
}


def _df_summary(df: pd.DataFrame, filename: str) -> str:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols = [c for c in df.columns if "date" in c.lower()]
    cat_cols = [c for c in df.select_dtypes(exclude="number").columns if c not in date_cols]

    lines = [
        f"File: {filename}",
        f"Rows: {len(df):,} | Columns: {len(df.columns)}",
        f"Columns: {', '.join(df.columns.tolist())}",
    ]
    if date_cols:
        for dc in date_cols:
            uniq = df[dc].dropna().unique()
            lines.append(f"Unique {dc} values ({len(uniq)}): {', '.join(sorted(str(v) for v in uniq))}")
    if numeric_cols:
        lines.append("\nNumeric column totals (all rows):")
        for col in numeric_cols:
            col_data = df[col].dropna()
            lines.append(
                f"  {col}: total={col_data.sum():,.4f}, mean={col_data.mean():,.4f}, "
                f"min={col_data.min():,.4f}, max={col_data.max():,.4f}, count={len(col_data):,}"
            )
    if date_cols and numeric_cols:
        primary_date = date_cols[0]
        lines.append(f"\nNumeric totals grouped by {primary_date}:")
        grouped = df.groupby(primary_date)[numeric_cols].sum()
        for date_val, row in grouped.iterrows():
            row_parts = ", ".join(f"{col}={row[col]:,.2f}" for col in numeric_cols)
            lines.append(f"  {date_val}: {row_parts}")
    if cat_cols:
        lines.append("\nCategorical column summaries:")
        for col in cat_cols[:10]:
            uniq = df[col].dropna().unique()
            sample = ", ".join(str(v) for v in uniq[:10])
            lines.append(f"  {col}: {len(uniq)} unique values — e.g. {sample}")
    lines.append(f"\nFirst 5 rows:\n{df.head(5).to_string(index=False)}")
    return "\n".join(lines)


@app.post("/upload")
async def upload_and_ask(
    file: UploadFile = File(...),
    question: str = Form(...),
    system_prompt: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    content_type = file.content_type or ""
    if not content_type or "/" not in content_type:
        content_type, _ = mimetypes.guess_type(file.filename or "")
        content_type = content_type or "application/octet-stream"
    if content_type == "image/jpg":
        content_type = "image/jpeg"
    raw = await file.read()

    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    parsed_df = None

    if content_type == "text/csv":
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="CSV file could not be decoded as UTF-8")
        parsed_df = await asyncio.to_thread(pd.read_csv, io.StringIO(text), low_memory=False)
        summary = await asyncio.to_thread(_df_summary, parsed_df, file.filename)
        user_content = [{"type": "text", "text": f"{summary}\n\nQuestion: {question}"}]

    elif content_type in EXCEL_TYPES:
        try:
            parsed_df = await asyncio.to_thread(pd.read_excel, io.BytesIO(raw), engine="openpyxl")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read Excel file: {e}")
        summary = await asyncio.to_thread(_df_summary, parsed_df, file.filename)
        user_content = [{"type": "text", "text": f"{summary}\n\nQuestion: {question}"}]

    elif content_type.startswith("text/"):
        try:
            text_content = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File could not be decoded as UTF-8 text")
        user_content = [{"type": "text", "text": f"File: {file.filename}\n\n{text_content}\n\n{question}"}]

    elif content_type in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        encoded = base64.standard_b64encode(raw).decode("utf-8")
        user_content = [
            {"type": "image", "source": {"type": "base64", "media_type": content_type, "data": encoded}},
            {"type": "text", "text": question},
        ]

    elif content_type == "application/pdf":
        encoded = base64.standard_b64encode(raw).decode("utf-8")
        user_content = [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": encoded}},
            {"type": "text", "text": question},
        ]

    else:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Supported: CSV, Excel, images (jpeg/png/gif/webp), PDF, and plain text.",
        )

    sid = session_id or str(uuid4())
    session = _get_or_create_session(
        sid, system_prompt or "You are a helpful assistant. Analyse the provided file and answer the user's question precisely."
    )

    if len(session["messages"]) >= SESSION_MAX_MESSAGES:
        first = session["messages"][:1]
        rest = session["messages"][1:][-(SESSION_MAX_MESSAGES - 2):]
        session["messages"] = first + rest

    session["file"] = file.filename
    if parsed_df is not None:
        session["df"] = parsed_df
        session["df_summary"] = summary

    session["messages"].append({"role": "user", "content": user_content})
    session["display"].append({"role": "user", "content": question})

    # Upload to Cloudinary in the background — don't block the response
    async def _upload_to_cloudinary():
        try:
            resource_type = "image" if content_type.startswith("image/") else "raw"
            result = await asyncio.to_thread(
                cloudinary.uploader.upload,
                io.BytesIO(raw),
                public_id=f"sessions/{sid}/{file.filename}",
                resource_type=resource_type,
                overwrite=True,
            )
            session["cloudinary_id"] = result["public_id"]
            session["cloudinary_url"] = result["secure_url"]
        except Exception as e:
            print(f"Cloudinary upload failed: {e}")

    asyncio.create_task(_upload_to_cloudinary())

    async def _generate():
        async for chunk in _stream_and_save(sid, model, 2048):
            yield chunk
        sessions[sid]["last_accessed"] = time()
        yield f"event: meta\ndata: {json.dumps(_session_meta(sid))}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    session = sessions.pop(session_id, None)
    if session:
        _delete_cloudinary_file(session)
    return {"cleared": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True, log_level="warning")
