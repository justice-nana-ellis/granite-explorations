import os
import base64
import io
import logging
import mimetypes
from pathlib import Path
import pandas as pd
from anthropic import AsyncAnthropic
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncIterator
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

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

class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None

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

async def _stream_chat(messages: list, system: Optional[str]) -> AsyncIterator[str]:
    kwargs = dict(model=chat_model, max_tokens=1024, messages=messages)
    if system:
        kwargs["system"] = system
    async with client.messages.stream(**kwargs) as stream:
        async for text in stream.text_stream:
            yield f"data: {text}\n\n"

@app.post("/chat")
async def chat(request: ChatRequest):
    messages = [{"role": "user", "content": request.message}]
    return StreamingResponse(
        _stream_chat(messages, request.system_prompt),
        media_type="text/event-stream",
    )

SUPPORTED_MEDIA_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "application/pdf",
    "text/plain", "text/csv", "text/html", "text/markdown",
}

@app.post("/upload", response_model=ChatResponse)
async def upload_and_ask(
    file: UploadFile = File(...),
    question: str = Form(...),
    system_prompt: Optional[str] = Form(None),
):
    content_type = file.content_type or ""
    if not content_type or "/" not in content_type:
        content_type, _ = mimetypes.guess_type(file.filename or "")
        content_type = content_type or "application/octet-stream"
    raw = await file.read()

    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Build the message content depending on file type
    if content_type == "text/csv":
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="CSV file could not be decoded as UTF-8")

        df = pd.read_csv(io.StringIO(text))

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        date_cols = [c for c in df.columns if "date" in c.lower()]
        cat_cols = [c for c in df.select_dtypes(exclude="number").columns if c not in date_cols]

        lines = [
            f"File: {file.filename}",
            f"Rows: {len(df):,} | Columns: {len(df.columns)}",
            f"Columns: {', '.join(df.columns.tolist())}",
        ]

        if date_cols:
            for dc in date_cols:
                uniq = df[dc].dropna().unique()
                lines.append(f"Unique {dc} values ({len(uniq)}): {', '.join(sorted(str(v) for v in uniq))}")

        if numeric_cols:
            lines.append("\nNumeric column totals:")
            for col in numeric_cols:
                col_data = df[col].dropna()
                lines.append(
                    f"  {col}: total={col_data.sum():,.4f}, mean={col_data.mean():,.4f}, "
                    f"min={col_data.min():,.4f}, max={col_data.max():,.4f}, count={len(col_data):,}"
                )

        if cat_cols:
            lines.append("\nCategorical column summaries:")
            for col in cat_cols[:10]:
                uniq = df[col].dropna().unique()
                sample = ", ".join(str(v) for v in uniq[:10])
                lines.append(f"  {col}: {len(uniq)} unique values — e.g. {sample}")

        lines.append(f"\nFirst 5 rows:\n{df.head(5).to_string(index=False)}")

        formatted = "\n".join(lines)
        user_content = [{"type": "text", "text": f"{formatted}\n\nQuestion: {question}"}]

    elif content_type.startswith("text/"):
        try:
            text_content = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File could not be decoded as UTF-8 text")

        user_content = [
            {
                "type": "text",
                "text": f"File: {file.filename}\n\n{text_content}\n\n{question}",
            }
        ]
    elif content_type in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        encoded = base64.standard_b64encode(raw).decode("utf-8")
        user_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": content_type,
                    "data": encoded,
                },
            },
            {"type": "text", "text": question},
        ]
    elif content_type == "application/pdf":
        encoded = base64.standard_b64encode(raw).decode("utf-8")
        user_content = [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": encoded,
                },
            },
            {"type": "text", "text": question},
        ]
    else:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Supported: images (jpeg/png/gif/webp), PDF, and plain text files.",
        )

    messages = [{"role": "user", "content": user_content}]
    system = system_prompt or "You are a helpful assistant. Analyse the provided file and answer the user's question precisely."

    async def _stream() -> AsyncIterator[str]:
        async with client.messages.stream(
            model=model,
            max_tokens=2048,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield f"data: {text}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True, log_level="warning")
