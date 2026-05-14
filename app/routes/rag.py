"""RAG endpoints — additive only. Existing /chat, /allChat, /upload are unchanged."""
from __future__ import annotations

import asyncio
import io
import logging
from typing import Optional
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.services import rag_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])


class RagQueryRequest(BaseModel):
    msg: str
    session_id: str
    top_k: int = 15
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: Optional[str] = None


@router.post("/ingest")
async def rag_ingest(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    """
    Upload a CSV or Excel file once. Every row is embedded and stored in Supabase pgvector.
    Returns a session_id you use for all subsequent /rag/query calls.
    No re-uploading needed — data persists in the database.
    Supported formats: .csv, .xlsx, .xls
    """
    fname = (file.filename or "").lower()
    if not any(fname.endswith(ext) for ext in (".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only .csv, .xlsx, and .xls files are supported.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="File is empty.")

    try:
        if fname.endswith(".csv"):
            df = await asyncio.to_thread(
                pd.read_csv, io.StringIO(raw.decode("utf-8")), low_memory=False
            )
        else:
            df = await asyncio.to_thread(
                pd.read_excel, io.BytesIO(raw), engine="openpyxl"
            )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"File parse error: {exc}")

    sid = session_id or str(uuid4())
    try:
        count = await rag_service.ingest_df(df, sid, file.filename)
    except Exception as exc:
        logger.exception("RAG ingest failed for session %s", sid)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return {
        "session_id": sid,
        "filename": file.filename,
        "rows_ingested": count,
        "message": f"Ingested {count} rows. Use session_id '{sid}' for all future queries.",
    }


@router.post("/query")
async def rag_query(body: RagQueryRequest):
    """
    Ask a question about your ingested data.
    Accepts JSON body with: msg, session_id, top_k, max_tokens, temperature, system_prompt.
    """
    if not body.msg.strip():
        raise HTTPException(status_code=400, detail="msg is required.")

    try:
        analysis, rows = await rag_service.rag_query(
            question=body.msg,
            session_id=body.session_id,
            top_k=body.top_k,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            system_prompt=body.system_prompt,
        )
    except Exception as exc:
        logger.exception("RAG query failed for session %s", body.session_id)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "session_id": body.session_id,
        "question": body.msg,
        "analysis": analysis,
        "retrieved_rows": len(rows),
        "sources": rows[:5],
    }


@router.delete("/session/{session_id}")
async def rag_delete_session(session_id: str):
    """Remove all ingested rows for a session from Supabase."""
    try:
        await rag_service.delete_session_data(session_id)
    except Exception as exc:
        logger.exception("RAG delete failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"session_id": session_id, "deleted": True}
