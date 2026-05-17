"""RAG endpoints — additive only. Existing /chat, /allChat, /upload are unchanged."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import re
from typing import Optional
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services import rag_service
from app.utils.prompt_library import RAG_FORECASTING_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


async def _prewarm_analytics(rag_id: str) -> None:
    """Background task: pre-compute analytics + context + charts after ingest.

    Uses default column map and 6-month horizon so the cache is warm before
    the first /forecast/full request arrives.
    """
    try:
        from app.config import settings
        from app.services.fund_forecast_service import (
            ColumnMap, build_analytics, build_chart_data,
            build_claude_context, cache_analytics, run_timesfm_forecasts,
        )
        rows = await rag_service.fetch_all_metadata(rag_id, limit=16000)
        if not rows:
            return
        df       = pd.DataFrame(rows)
        col      = ColumnMap()
        horizon  = 6
        analytics = await asyncio.to_thread(build_analytics, df, col)
        context   = await asyncio.to_thread(build_claude_context, analytics, col, horizon)

        timesfm_results = None
        if settings.use_timesfm:
            timesfm_results = await run_timesfm_forecasts(analytics, horizon)

        charts = await asyncio.to_thread(build_chart_data, analytics, timesfm_results, horizon)
        cache_analytics([rag_id], horizon, col, {
            "source_rows":     len(rows),
            "analytics":       analytics,
            "context":         context,
            "charts":          charts,
            "timesfm_results": timesfm_results,
        })
        logger.info("Analytics pre-warmed for rag_id=%s (%d rows, timesfm=%s)",
                    rag_id, len(rows), timesfm_results is not None)
    except Exception as exc:
        logger.warning("Analytics pre-warm failed for rag_id=%s: %s", rag_id, exc)

# ── Forecast tool schema ────────────────────────────────────────────────────────
# Claude is forced to call this tool, so its input is always a validated dict.
# No JSON parsing or text extraction needed.

FORECAST_TOOL: dict = {
    "name": "submit_forecast",
    "description": (
        "Submit the complete quantitative forecasting report populated from the provided data. "
        "Every field must be filled. Leave no section empty."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "executive_summary": {
                "type": "string",
                "description": "4-6 sentence high-level outlook leading with the most critical insight.",
            },
            "data_period": {
                "type": "object",
                "properties": {
                    "earliest":         {"type": "string"},
                    "latest":           {"type": "string"},
                    "months_of_history":{"type": "integer"},
                },
                "required": ["earliest", "latest", "months_of_history"],
            },
            "current_snapshot": {
                "type": "object",
                "description": "Current AUM, currency, portfolio/fund counts, latest net flow, key metrics list.",
                "additionalProperties": True,
            },
            "forecasts": {
                "type": "object",
                "description": "All forecast sections: aum_trend, net_flows, revenue, churn_risk, portfolio_mix_drift. Each must include projections array and chart_data.",
                "properties": {
                    "aum_trend":          {"type": "object", "additionalProperties": True},
                    "net_flows":          {"type": "object", "additionalProperties": True},
                    "revenue":            {"type": "object", "additionalProperties": True},
                    "churn_risk":         {"type": "object", "additionalProperties": True},
                    "portfolio_mix_drift":{"type": "object", "additionalProperties": True},
                },
                "required": ["aum_trend", "net_flows", "revenue", "churn_risk", "portfolio_mix_drift"],
            },
            "top_10_performers": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
                "description": "Top 10 funds/clients/portfolios by AUM growth and flow momentum with forward projections.",
            },
            "top_10_at_risk": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
                "description": "Top 10 funds/clients/portfolios most at risk of AUM decline or redemption.",
            },
            "strategic_recommendations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Minimum 5 specific, actionable recommendations referencing exact figures from the data.",
            },
            "methodology": {
                "type": "string",
                "description": "Forecasting methodology, data coverage, and key assumptions used.",
            },
            "data_warnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Data quality issues, missing fields, or assumptions made (e.g. assumed fee rate).",
            },
        },
        "required": [
            "executive_summary", "data_period", "current_snapshot", "forecasts",
            "top_10_performers", "top_10_at_risk", "strategic_recommendations",
            "methodology", "data_warnings",
        ],
    },
}

router = APIRouter(prefix="/rag", tags=["rag"])


def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Event line."""
    return f"data: {json.dumps(data)}\n\n"


def _repair_json(s: str) -> str:
    """Fix the most common LLM JSON mistakes before parsing."""
    # trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Python-style True/False/None
    s = re.sub(r"\bTrue\b",  "true",  s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b",  "null",  s)
    return s


def _find_json_object(text: str) -> str:
    """Return the outermost {...} block using depth-aware scanning."""
    start = text.index("{")
    depth, in_str, escape = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    raise ValueError("No complete JSON object found in response.")


def _extract_json(text: str) -> dict:
    """Parse JSON from Claude's response with multi-stage fallback and auto-repair."""
    text = text.strip()

    # 1 — direct parse (ideal path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2 — strip markdown fences
    for pattern in [r"```json\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            candidate = m.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    return json.loads(_repair_json(candidate))
                except json.JSONDecodeError:
                    pass

    # 3 — depth-aware brace extraction + repair
    try:
        candidate = _find_json_object(text)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return json.loads(_repair_json(candidate))
    except (ValueError, json.JSONDecodeError):
        pass

    logger.warning(
        "RAG forecast JSON parse failed. Response preview: %.500s", text
    )
    raise ValueError("Could not parse JSON from model response.")


def _build_forecast_message(msg: str, horizon: str, context: str) -> str:
    base = (
        "Produce a complete, comprehensive forecasting report covering EVERY section "
        "of the schema without exception: AUM trend, net flows, revenue, churn risk, "
        "portfolio mix drift, top 10 performers, top 10 at risk, strategic recommendations, "
        "and methodology. Leave no section empty. Use all available data rows."
    )
    extra = msg.strip()
    default_msg = "Produce a full forecasting report on all available data."
    additional = (
        f"\n\nADDITIONAL USER FOCUS: {extra}"
        if extra and extra != default_msg
        else ""
    )
    return (
        f"[FORECAST DIRECTIVE]\n{base}{additional}\n\n"
        f"[FORECAST HORIZON]\nProject forward: {horizon} from the latest available data point.\n\n"
        f"[DATA ROWS — retrieved by semantic similarity]\n{context}"
    )


# ── Request models ─────────────────────────────────────────────────────────────

class RagQueryRequest(BaseModel):
    msg: str = "Produce a full forecasting report on all available data."
    rag_ids: list[str]
    forecast_horizon: str = "6m"   # e.g. "3m", "6m", "1y", "2y"
    top_k: int = 100               # high to capture maximum historical data
    max_tokens: int = 16384
    system_prompt: Optional[str] = None


class RagChatRequest(BaseModel):
    msg: str = "Produce a full forecasting report on all available data."
    rag_ids: list[str]
    forecast_horizon: str = "6m"
    top_k: int = 100
    max_tokens: int = 16384
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

            # Pre-warm analytics cache so /forecast/full skips computation
            if results:
                asyncio.create_task(_prewarm_analytics(rid))

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


# ── Query — forecasting agent, plain JSON response ─────────────────────────────

@router.post("/query")
async def rag_query(body: RagQueryRequest):
    """
    Forecasting agent. Uses tool use to guarantee a valid structured response —
    no JSON parsing, no parse_error. Returns the full forecast object directly.
    """
    if not body.rag_ids:
        raise HTTPException(status_code=400, detail="At least one rag_id is required.")

    from app.services.claude_service import claude_service
    from app.config import settings

    try:
        all_rows, context = await rag_service.rag_retrieve_multi(
            question=body.msg,
            rag_ids=body.rag_ids,
            top_k=body.top_k,
        )
    except Exception as exc:
        logger.exception("RAG forecast retrieve failed for rag_ids %s", body.rag_ids)
        raise HTTPException(status_code=500, detail=str(exc))

    system   = body.system_prompt or RAG_FORECASTING_SYSTEM_PROMPT
    messages = [{"role": "user", "content": _build_forecast_message(body.msg, body.forecast_horizon, context)}]

    try:
        forecast = await claude_service.complete_with_tool(
            messages=messages,
            system=system,
            tool=FORECAST_TOOL,
            model=settings.claude_model,
            max_tokens=body.max_tokens,
        )
    except Exception as exc:
        logger.exception("RAG forecast tool call failed for rag_ids %s", body.rag_ids)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "rag_ids":          body.rag_ids,
        "forecast_horizon": body.forecast_horizon,
        "retrieved_rows":   len(all_rows),
        "forecast":         forecast,
    }


# ── Chat — forecasting agent, streaming SSE ────────────────────────────────────

@router.post("/chat")
async def rag_chat(body: RagChatRequest):
    """
    Forecasting agent: streams the forecast as it is generated, then emits a
    final 'forecast' event containing the fully parsed structured JSON object
    (including all chart data) so the frontend can render visuals immediately.

    Event types:
      sources   — fired first: retrieved row count (no waiting)
      chunk     — one text fragment of the forecast as Claude writes it
      forecast  — parsed structured JSON object with all chart-ready data
      done      — stream complete
      error     — something failed mid-stream
    """
    if not body.rag_ids:
        raise HTTPException(status_code=400, detail="At least one rag_id is required.")

    from app.services.claude_service import claude_service
    from app.config import settings

    try:
        all_rows, context = await rag_service.rag_retrieve_multi(
            question=body.msg,
            rag_ids=body.rag_ids,
            top_k=body.top_k,
        )
    except Exception as exc:
        logger.exception("RAG forecast retrieve failed for rag_ids %s", body.rag_ids)
        raise HTTPException(status_code=500, detail=str(exc))

    system   = body.system_prompt or RAG_FORECASTING_SYSTEM_PROMPT
    messages = [{"role": "user", "content": _build_forecast_message(body.msg, body.forecast_horizon, context)}]

    async def event_stream():
        # Emit sources immediately so the client knows retrieval succeeded
        yield _sse({
            "type":             "sources",
            "rag_ids":          body.rag_ids,
            "forecast_horizon": body.forecast_horizon,
            "retrieved_rows":   len(all_rows),
            "message":          f"Retrieved {len(all_rows):,} data rows. Generating forecast…",
        })

        # Tool-use call (blocking but guaranteed valid) — run via task so SSE stays open
        queue: asyncio.Queue = asyncio.Queue()

        async def run_forecast():
            try:
                result = await claude_service.complete_with_tool(
                    messages=messages,
                    system=system,
                    tool=FORECAST_TOOL,
                    model=settings.claude_model,
                    max_tokens=body.max_tokens,
                )
                await queue.put({"ok": True, "forecast": result})
            except Exception as exc:
                await queue.put({"ok": False, "error": str(exc)})

        asyncio.create_task(run_forecast())

        # Keep the SSE connection alive with heartbeats while waiting
        import asyncio as _asyncio
        heartbeat = 0
        while True:
            try:
                result = queue.get_nowait()
                break
            except _asyncio.QueueEmpty:
                heartbeat += 1
                yield _sse({"type": "heartbeat", "tick": heartbeat,
                            "message": "Forecast generation in progress…"})
                await _asyncio.sleep(3)

        if not result["ok"]:
            logger.exception("RAG forecast tool call failed for rag_ids %s", body.rag_ids)
            yield _sse({"type": "error", "message": result["error"]})
            return

        yield _sse({"type": "forecast", "forecast": result["forecast"]})
        yield _sse({
            "type":             "done",
            "rag_ids":          body.rag_ids,
            "forecast_horizon": body.forecast_horizon,
            "retrieved_rows":   len(all_rows),
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


@router.get("/{rag_id}")
async def rag_list_files(rag_id: str):
    """Return the RAG and all its ingested files with row counts."""
    try:
        files = await rag_service.list_rag_files(rag_id)
    except Exception as exc:
        logger.exception("RAG list files failed for rag_id %s", rag_id)
        raise HTTPException(status_code=500, detail=str(exc))

    total_rows = sum(f.get("row_count", 0) for f in files)
    return {
        "rag_id": rag_id,
        "total_files": len(files),
        "total_rows": total_rows,
        "files": files,
        "message": (
            f"RAG '{rag_id}' contains {len(files)} file(s) with {total_rows:,} total rows."
            if files else
            f"RAG '{rag_id}' has no ingested files."
        ),
    }


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
