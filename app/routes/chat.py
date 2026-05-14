from __future__ import annotations

import asyncio
import json
import logging
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from typing import Optional

from app.config import settings
from app.models.requests import ChatRequest
from app.services.claude_service import claude_service
from app.services.file_service import EXCEL_TYPES, IMAGE_TYPES, WORD_TYPES, PPTX_TYPES, normalize_content_type
from app.services.session_service import session_service
from app.services.storage_service import storage_service
from app.services.visualization_service import visualization_service
from app.utils.html_utils import analysis_to_html
from app.utils.prompt_library import ANALYSIS_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(request: ChatRequest):
    sid = request.session_id or str(uuid4())
    session = session_service.get_or_create(sid, request.system_prompt or CHAT_SYSTEM_PROMPT)
    session_service.trim_messages(session)

    if session.get("df") is None and session.get("cloudinary_url"):
        await session_service.reload_df(session)

    if session.get("df_summary"):
        msg = f"[File context]\n{session['df_summary']}\n\n[Question]\n{request.message}"
    else:
        msg = request.message

    session["messages"].append({"role": "user", "content": msg})
    session["display"].append({"role": "user", "content": request.message})

    async def _generate():
        full_reply: list[str] = []
        async for chunk in claude_service.stream(
            session["messages"][:-1] + [session["messages"][-1]],
            system=session["system"],
            model=settings.claude_chat_model,
            max_tokens=1024,
        ):
            full_reply.append(chunk)
            yield f"data: {chunk}\n\n"

        reply_text = "".join(full_reply)
        session["messages"].append({"role": "assistant", "content": reply_text})
        session["display"].append({"role": "assistant", "content": reply_text})
        session_service.touch(sid)
        meta = session_service.meta(sid)
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.post("/allChat")
async def all_chat(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    files: Optional[list[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
    http_request: Request = None,
):
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    sid = session_id or str(uuid4())
    session = session_service.get_or_create(sid, system_prompt or ANALYSIS_SYSTEM_PROMPT)
    session_service.trim_messages(session)

    if file and (not files or len(files) == 0):
        files = [file]

    filename_display = session.get("file") or "Chat Session"
    msg_content: object = question

    if files and len(files) > 0:
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 files per request")

        file_summaries: list[str] = []
        primary_df = None

        for idx, f in enumerate(files):
            if not f.filename:
                continue
            content_type = normalize_content_type(f)
            raw = await f.read()
            if not raw:
                file_summaries.append(f"[File {idx + 1}: {f.filename}] — empty, skipped")
                continue

            parsed_df = None
            summary = None

            try:
                import asyncio as _asyncio
                import io as _io
                import pandas as _pd
                from app.services.file_service import _extract_docx, _extract_pptx
                from app.utils.df_utils import df_summary as _df_summary

                if content_type == "text/csv":
                    text = raw.decode("utf-8")
                    parsed_df = await _asyncio.to_thread(_pd.read_csv, _io.StringIO(text), low_memory=False)
                    summary = await _asyncio.to_thread(_df_summary, parsed_df, f.filename)
                elif content_type in EXCEL_TYPES:
                    parsed_df = await _asyncio.to_thread(_pd.read_excel, _io.BytesIO(raw), engine="openpyxl")
                    summary = await _asyncio.to_thread(_df_summary, parsed_df, f.filename)
                elif content_type in WORD_TYPES:
                    text_content = await _asyncio.to_thread(_extract_docx, raw)
                    summary = f"[File {idx + 1}: {f.filename} (DOCX)]\n{text_content[:1500]}"
                elif content_type in PPTX_TYPES:
                    text_content = await _asyncio.to_thread(_extract_pptx, raw)
                    summary = f"[File {idx + 1}: {f.filename} (PPTX)]\n{text_content[:1500]}"
                elif content_type.startswith("text/"):
                    text_content = raw.decode("utf-8")
                    summary = f"[File {idx + 1}: {f.filename} (Text)]\n{text_content[:1500]}"
                elif content_type in IMAGE_TYPES:
                    summary = f"[File {idx + 1}: {f.filename}] Image file ({content_type})"
                elif content_type == "application/pdf":
                    summary = f"[File {idx + 1}: {f.filename}] PDF document"
                else:
                    summary = f"[File {idx + 1}: {f.filename}] Unsupported type: {content_type}"
            except Exception as exc:
                summary = f"[File {idx + 1}: {f.filename}] Error: {str(exc)[:100]}"

            if summary:
                file_summaries.append(summary)

            if parsed_df is not None and primary_df is None:
                import asyncio as _asyncio
                from app.utils.df_utils import df_summary as _df_summary
                primary_df = parsed_df
                session["df"] = parsed_df
                session["df_summary"] = summary
                session["file"] = f.filename
                session["file_content_type"] = content_type
                session["file_raw"] = raw
            elif primary_df is None:
                session["file"] = f.filename
                session["file_content_type"] = content_type
                session["file_raw"] = raw
                if content_type in WORD_TYPES or content_type in PPTX_TYPES or content_type.startswith("text/"):
                    session["file_summary"] = summary

            async def _bg_upload(raw=raw, fname=f.filename, ctype=content_type):
                rtype = "image" if ctype.startswith("image/") else "raw"
                try:
                    result = await storage_service.upload_file(
                        raw,
                        public_id=f"sessions/{sid}/{fname}",
                        resource_type=rtype,
                    )
                    if session.get("file") == fname:
                        session["cloudinary_id"] = result["public_id"]
                        session["cloudinary_url"] = result["secure_url"]
                except Exception as exc:
                    logger.error("Cloudinary upload failed for %s: %s", fname, exc)

            asyncio.create_task(_bg_upload())

        combined = "\n\n".join(file_summaries) if file_summaries else "Files processed."
        msg_content = f"[Uploaded Files]\n{combined}\n\n[User Question]\n{question}"
        filename_display = f"{len(files)} file(s)"

    else:
        if session.get("df") is None and session.get("cloudinary_url"):
            await session_service.reload_df(session)

        ct = session.get("file_content_type", "")
        import base64
        if session.get("df_summary"):
            msg_content = f"[File context]\n{session['df_summary']}\n\n[Question]\n{question}"
        elif ct in IMAGE_TYPES and session.get("file_raw"):
            enc = base64.standard_b64encode(session["file_raw"]).decode("utf-8")
            msg_content = [
                {"type": "image", "source": {"type": "base64", "media_type": ct, "data": enc}},
                {"type": "text", "text": question},
            ]
        elif ct == "application/pdf" and session.get("file_raw"):
            enc = base64.standard_b64encode(session["file_raw"]).decode("utf-8")
            msg_content = [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": enc}},
                {"type": "text", "text": question},
            ]
        elif session.get("file_summary"):
            msg_content = f"[File context]\n{session['file_summary']}\n\n[Question]\n{question}"
        else:
            msg_content = question
        filename_display = session.get("file") or "Chat Session"

    session["messages"].append({"role": "user", "content": msg_content})
    session["display"].append({"role": "user", "content": question})

    async def _analysis():
        return await claude_service.complete(
            session["messages"],
            system=session["system"],
            model=settings.claude_chat_model,
            max_tokens=1024,
        )

    async def _visuals():
        if session.get("df") is None:
            return [], None
        results = await asyncio.gather(
            visualization_service.build_artifacts(sid, session, max_charts=3),
            visualization_service.build_forecast(sid, session),
            return_exceptions=True,
        )
        arts = results[0] if not isinstance(results[0], Exception) else []
        fore = results[1] if not isinstance(results[1], Exception) else None
        return arts, fore

    (analysis_text, (artifacts, forecast)) = await asyncio.gather(_analysis(), _visuals())
    session["messages"].append({"role": "assistant", "content": analysis_text})
    session["display"].append({"role": "assistant", "content": analysis_text})

    session_service.touch(sid)

    base_url = str(http_request.base_url).rstrip("/") if http_request else "http://localhost:8000"
    wants_json = (response_format or "").lower() in {"json", "bundle"}

    if wants_json:
        return {
            "session_id": sid,
            "file": session.get("file"),
            "analysis": analysis_text,
            "artifacts": artifacts,
            "forecast": forecast,
            "links": {
                "dashboard": f"{base_url}/dashboard/{sid}/render",
                "artifacts": f"{base_url}/artifacts/{sid}",
                "forecast": f"{base_url}/forecast/{sid}",
                "visualize": f"{base_url}/visualize/{sid}",
            },
        }

    from app.utils.html_utils import templates
    kpi_html = visualization_service.build_kpi_cards_html(session)
    cjs_html, cjs_init = visualization_service.build_chartjs_html(session)
    jsx_html = visualization_service.build_jsx_snippet(session, analysis_text) if session.get("df") is not None else ""

    return HTMLResponse(content=templates.get_template("allchat.html").render({
        "request": http_request,
        "session_id": sid,
        "filename": filename_display,
        "question": question,
        "analysis_html": analysis_to_html(analysis_text),
        "kpi_html": kpi_html,
        "chartjs_html": cjs_html,
        "chart_init_js": cjs_init,
        "artifacts": artifacts,
        "forecast": forecast,
        "jsx_html": jsx_html,
        "dashboard_url": f"{base_url}/dashboard/{sid}/render",
        "artifacts_url": f"{base_url}/artifacts/{sid}",
        "forecast_url": f"{base_url}/forecast/{sid}",
        "visualize_url": f"{base_url}/visualize/{sid}",
    }))
