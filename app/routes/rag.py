"""RAG endpoints — additive only. Existing /chat, /allChat, /upload are unchanged."""
from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import Optional
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services import rag_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])


def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Event line."""
    return f"data: {json.dumps(data)}\n\n"


# ── Request models ─────────────────────────────────────────────────────────────

class RagQueryRequest(BaseModel):
    msg: str
    rag_id: str
    top_k: int = 15
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: Optional[str] = None


class RagChatRequest(BaseModel):
    msg: str
    rag_ids: list[str]
    top_k: int = 10          # rows retrieved per rag_id
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: Optional[str] = None


# ── Ingest ─────────────────────────────────────────────────────────────────────

@router.post("/ingest")
async def rag_ingest(
    files: Optional[list[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
    rag_id: Optional[str] = Form(None),
):
    """
    Upload one or multiple CSV/Excel files.
    Returns a Server-Sent Events stream (text/event-stream).

    Each event is a JSON object with a 'type' field:
      start      — ingestion has begun, includes rag_id
      info       — file parsed, row count known, embedding starting
      progress   — batch checkpoint (N / total rows ingested)
      file_done  — one file finished successfully
      error      — one file failed (others continue)
      done       — all files finished, final summary included

    The terminal also receives all the same messages via the normal logger.
    """
    all_files: list[UploadFile] = list(files or [])
    if file:
        all_files = [file] + all_files
    if not all_files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    rid = rag_id or str(uuid4())

    # ── Parse every uploaded file before the stream opens ──────────────────────
    # (request body must be fully read before we start streaming the response)
    parsed: list[tuple[str, pd.DataFrame]] = []
    parse_errors: list[dict] = []

    for f in all_files:
        fname = (f.filename or "").lower()
        if not any(fname.endswith(ext) for ext in (".csv", ".xlsx", ".xls")):
            parse_errors.append({
                "filename": f.filename,
                "error": "Unsupported format — use .csv, .xlsx, or .xls",
            })
            continue

        raw = await f.read()
        if not raw:
            parse_errors.append({"filename": f.filename, "error": "File is empty"})
            continue

        try:
            if fname.endswith(".csv"):
                df = await asyncio.to_thread(
                    pd.read_csv, io.StringIO(raw.decode("utf-8")), low_memory=False
                )
            else:
                df = await asyncio.to_thread(
                    pd.read_excel, io.BytesIO(raw), engine="openpyxl"
                )
            parsed.append((f.filename, df))
        except Exception as exc:
            parse_errors.append({"filename": f.filename, "error": f"Parse error: {exc}"})

    if not parsed:
        raise HTTPException(
            status_code=400,
            detail={"message": "All files failed to parse", "errors": parse_errors},
        )

    # ── Stream ingestion progress as SSE ───────────────────────────────────────
    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        results: list[dict] = []
        ingest_errors: list[dict] = list(parse_errors)

        async def run_ingestion():
            for fname, df in parsed:
                row_count = len(df)
                await queue.put({
                    "type": "info",
                    "filename": fname,
                    "row_count": row_count,
                    "message": f"[{fname}] Parsed — {row_count:,} rows. Starting embedding and storage…",
                })
                try:
                    async def on_progress(msg: str, q: asyncio.Queue = queue):
                        await q.put({"type": "progress", "message": msg})

                    count = await rag_service.ingest_df(
                        df, rid, fname, progress_callback=on_progress
                    )
                    results.append({"filename": fname, "rows_ingested": count})
                    await queue.put({
                        "type": "file_done",
                        "filename": fname,
                        "rows_ingested": count,
                        "message": f"[{fname}] Complete — {count:,} rows stored successfully.",
                    })
                except Exception as exc:
                    logger.exception("RAG ingest failed for file %s rag_id %s", fname, rid)
                    ingest_errors.append({"filename": fname, "error": str(exc)})
                    await queue.put({
                        "type": "error",
                        "filename": fname,
                        "message": f"[{fname}] Failed — {exc}",
                    })

            total_ingested = sum(r["rows_ingested"] for r in results)
            await queue.put({
                "type": "done",
                "rag_id": rid,
                "files_ingested": len(results),
                "total_rows_ingested": total_ingested,
                "files": results,
                "errors": ingest_errors,
                "message": (
                    f"Ingestion complete — {len(results)} file(s), {total_ingested:,} rows "
                    f"stored under rag_id '{rid}'. Use this rag_id to query your data."
                ),
            })
            await queue.put(None)  # sentinel

        asyncio.create_task(run_ingestion())

        yield _sse({
            "type": "start",
            "rag_id": rid,
            "files_to_ingest": len(parsed),
            "message": f"Starting ingestion of {len(parsed)} file(s) into rag_id '{rid}'…",
        })

        while True:
            item = await queue.get()
            if item is None:
                break
            yield _sse(item)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Query (single rag_id) ──────────────────────────────────────────────────────

@router.post("/query")
async def rag_query(body: RagQueryRequest):
    """Ask a question against a single rag_id."""
    if not body.msg.strip():
        raise HTTPException(status_code=400, detail="msg is required.")

    try:
        analysis, rows = await rag_service.rag_query(
            question=body.msg,
            rag_id=body.rag_id,
            top_k=body.top_k,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            system_prompt=body.system_prompt,
        )
    except Exception as exc:
        logger.exception("RAG query failed for rag_id %s", body.rag_id)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "rag_id": body.rag_id,
        "question": body.msg,
        "analysis": analysis,
        "retrieved_rows": len(rows),
        "sources": rows[:5],
    }


# ── Chat (multiple rag_ids, streaming) ────────────────────────────────────────

@router.post("/chat")
async def rag_chat(body: RagChatRequest):
    """
    Ask a question across multiple rag_ids. Streams the answer as SSE.

    Event types:
      sources  — fired once upfront: retrieved row count + top-5 source rows
      chunk    — one text fragment of Claude's answer (reassemble in order)
      error    — Claude call failed
      done     — stream finished
    """
    if not body.msg.strip():
        raise HTTPException(status_code=400, detail="msg is required.")
    if not body.rag_ids:
        raise HTTPException(status_code=400, detail="At least one rag_id is required.")

    # Retrieve rows before opening the stream so errors surface as HTTP 500
    try:
        all_rows, context = await rag_service.rag_retrieve_multi(
            question=body.msg,
            rag_ids=body.rag_ids,
            top_k=body.top_k,
        )
    except Exception as exc:
        logger.exception("RAG retrieve failed for rag_ids %s", body.rag_ids)
        raise HTTPException(status_code=500, detail=str(exc))

    system = body.system_prompt or (
        "You are a precise data analyst. Answer using only the provided data rows. "
        "Be specific, cite exact numbers, and highlight key trends."
    )
    messages = [{
        "role": "user",
        "content": f"[Relevant data rows]\n{context}\n\n[Question]\n{body.msg}",
    }]

    async def event_stream():
        from app.services.claude_service import claude_service

        # Send retrieved sources first so the client can render them immediately
        yield _sse({
            "type": "sources",
            "rag_ids": body.rag_ids,
            "question": body.msg,
            "retrieved_rows": len(all_rows),
            "sources": all_rows[:5],
        })

        try:
            async for chunk in claude_service.stream(
                messages=messages,
                system=system,
                model=None,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
            ):
                yield _sse({"type": "chunk", "text": chunk})
        except Exception as exc:
            logger.exception("RAG chat stream failed for rag_ids %s", body.rag_ids)
            yield _sse({"type": "error", "message": str(exc)})
            return

        yield _sse({
            "type": "done",
            "rag_ids": body.rag_ids,
            "retrieved_rows": len(all_rows),
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Management ─────────────────────────────────────────────────────────────────

@router.get("/ragIDs")
async def rag_list_ids():
    """List every rag_id with its files and total row count."""
    try:
        sessions = await rag_service.list_all_rags()
    except Exception as exc:
        logger.exception("RAG list ragIDs failed")
        raise HTTPException(status_code=500, detail=str(exc))
    # rename session_id → rag_id in the response
    for s in sessions:
        s["rag_id"] = s.pop("session_id", s.get("rag_id"))
    return {"rag_ids": sessions, "count": len(sessions)}


@router.get("/{rag_id}/files")
async def rag_list_files(rag_id: str):
    """List all files inside a rag_id with row counts."""
    try:
        files = await rag_service.list_rag_files(rag_id)
    except Exception as exc:
        logger.exception("RAG list files failed for rag_id %s", rag_id)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"rag_id": rag_id, "files": files, "count": len(files)}


@router.delete("/{rag_id}/file/{filename}")
async def rag_delete_file(rag_id: str, filename: str):
    """
    Remove a single file from a rag_id. Streams deletion progress as SSE.

    Event types:
      start     — deletion beginning
      progress  — batch of rows deleted (Supabase only; one event per 200 rows)
      done      — finished, includes total rows_deleted
      error     — something went wrong
    """
    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def run_delete():
            try:
                async def on_progress(deleted_so_far: int, msg: str, q: asyncio.Queue = queue):
                    await q.put({"type": "progress", "deleted_so_far": deleted_so_far, "message": msg})

                count = await rag_service.delete_rag_file(
                    rag_id, filename, progress_callback=on_progress
                )
                await queue.put({
                    "type": "done",
                    "rag_id": rag_id,
                    "filename": filename,
                    "rows_deleted": count,
                    "message": (
                        f"File '{filename}' has been successfully deleted from rag_id '{rag_id}'. "
                        f"{count:,} rows removed."
                    ),
                })
            except Exception as exc:
                logger.exception("RAG delete file failed rag_id=%s file=%s", rag_id, filename)
                await queue.put({"type": "error", "message": str(exc)})
            await queue.put(None)

        asyncio.create_task(run_delete())

        yield _sse({
            "type": "start",
            "rag_id": rag_id,
            "filename": filename,
            "message": f"Starting deletion of '{filename}' from rag_id '{rag_id}'…",
        })

        while True:
            item = await queue.get()
            if item is None:
                break
            yield _sse(item)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.delete("/{rag_id}")
async def rag_delete(rag_id: str):
    """
    Delete an entire rag_id and all its files. Streams deletion progress as SSE.

    Event types:
      start     — deletion beginning
      progress  — batch of rows deleted (Supabase only; one event per 200 rows)
      done      — finished
      error     — something went wrong
    """
    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def run_delete():
            try:
                async def on_progress(deleted_so_far: int, msg: str, q: asyncio.Queue = queue):
                    await q.put({"type": "progress", "deleted_so_far": deleted_so_far, "message": msg})

                await rag_service.delete_rag_data(rag_id, progress_callback=on_progress)
                await queue.put({
                    "type": "done",
                    "rag_id": rag_id,
                    "deleted": True,
                    "message": f"RAG '{rag_id}' and all its files have been successfully deleted.",
                })
            except Exception as exc:
                logger.exception("RAG delete failed for rag_id %s", rag_id)
                await queue.put({"type": "error", "message": str(exc)})
            await queue.put(None)

        asyncio.create_task(run_delete())

        yield _sse({
            "type": "start",
            "rag_id": rag_id,
            "message": f"Starting deletion of rag_id '{rag_id}'…",
        })

        while True:
            item = await queue.get()
            if item is None:
                break
            yield _sse(item)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
