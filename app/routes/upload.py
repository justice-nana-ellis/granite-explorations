import asyncio
import json
import logging
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.services.claude_service import claude_service
from app.services.file_service import parse_upload, normalize_content_type
from app.services.session_service import session_service
from app.services.storage_service import storage_service
from app.services.visualization_service import visualization_service
from app.utils.html_utils import analysis_to_html
from app.utils.prompt_library import ANALYSIS_SYSTEM_PROMPT, FILE_ANALYSIS_SYSTEM_PROMPT

logger = logging.getLogger(__name__)
router = APIRouter(tags=["upload"])


@router.post("/upload")
async def upload_and_ask(
    file: UploadFile = File(...),
    question: str = Form(...),
    system_prompt: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    parsed = await parse_upload(file, question)
    sid = session_id or str(uuid4())
    session = session_service.get_or_create(
        sid, system_prompt or FILE_ANALYSIS_SYSTEM_PROMPT
    )
    session_service.trim_messages(session)

    session["file"] = parsed.filename
    if parsed.df is not None:
        session["df"] = parsed.df
        session["df_summary"] = parsed.df_summary_text
    if parsed.file_raw is not None:
        session["file_raw"] = parsed.file_raw
        session["file_content_type"] = parsed.content_type

    session["messages"].append({"role": "user", "content": parsed.user_content})
    session["display"].append({"role": "user", "content": question})

    async def _bg_upload():
        resource_type = "image" if parsed.content_type.startswith("image/") else "raw"
        try:
            result = await storage_service.upload_file(
                parsed.raw,
                public_id=f"sessions/{sid}/{parsed.filename}",
                resource_type=resource_type,
            )
            session["cloudinary_id"] = result["public_id"]
            session["cloudinary_url"] = result["secure_url"]
        except Exception as exc:
            logger.error("Cloudinary upload failed: %s", exc)
        await storage_service.upload_state(sid, {
            "file": session.get("file"),
            "cloudinary_id": session.get("cloudinary_id"),
            "cloudinary_url": session.get("cloudinary_url"),
            "system": session.get("system"),
            "display": session.get("display", []),
        })

    asyncio.create_task(_bg_upload())

    async def _generate():
        full_reply: list[str] = []
        async for chunk in claude_service.stream(
            session["messages"],
            system=session["system"],
            model=settings.claude_model,
            max_tokens=2048,
        ):
            full_reply.append(chunk)
            yield f"data: {chunk}\n\n"

        reply_text = "".join(full_reply)
        session["messages"].append({"role": "assistant", "content": reply_text})
        session["display"].append({"role": "assistant", "content": reply_text})
        session_service.touch(sid)
        yield f"event: meta\ndata: {json.dumps(session_service.meta(sid))}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.post("/upload-file")
async def upload_file_only(
    file: UploadFile = File(...),
    system_prompt: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    parsed = await parse_upload(file, "")
    sid = session_id or str(uuid4())
    session = session_service.get_or_create(sid, system_prompt or ANALYSIS_SYSTEM_PROMPT)

    session["file"] = parsed.filename
    session["file_content_type"] = parsed.content_type
    session["file_summary"] = parsed.file_summary
    if parsed.df is not None:
        session["df"] = parsed.df
        session["df_summary"] = parsed.df_summary_text
    if parsed.file_raw is not None:
        session["file_raw"] = parsed.file_raw
    session_service.touch(sid)

    async def _bg_upload():
        resource_type = "image" if parsed.content_type.startswith("image/") else "raw"
        try:
            result = await storage_service.upload_file(
                parsed.raw,
                public_id=f"sessions/{sid}/{parsed.filename}",
                resource_type=resource_type,
            )
            session["cloudinary_id"] = result["public_id"]
            session["cloudinary_url"] = result["secure_url"]
        except Exception as exc:
            logger.error("Cloudinary upload failed: %s", exc)
        await storage_service.upload_state(sid, {
            "file": session.get("file"),
            "cloudinary_id": session.get("cloudinary_id"),
            "cloudinary_url": session.get("cloudinary_url"),
            "system": session.get("system"),
            "display": session.get("display", []),
        })

    asyncio.create_task(_bg_upload())

    return {
        "session_id": sid,
        "file": parsed.filename,
        "content_type": parsed.content_type,
        "size_bytes": len(parsed.raw),
        "rows": len(parsed.df) if parsed.df is not None else None,
        "columns": list(parsed.df.columns) if parsed.df is not None else None,
        "summary_preview": (parsed.df_summary_text or parsed.file_summary or "")[:300],
        "status": "ready",
        "next": f'POST /analyze  {{"session_id": "{sid}", "message": "your question here"}}',
    }


@router.get("/upload-html")
def upload_html_form():
    from app.utils.html_utils import templates
    from fastapi import Request

    class _FakeRequest:
        pass

    return HTMLResponse(
        templates.get_template("upload_form.html").render({"request": _FakeRequest()})
    )


@router.post("/upload-html")
async def upload_and_render_html(
    file: UploadFile = File(...),
    question: str = Form("Create clickable visual artifacts and a forecast from this file"),
    system_prompt: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    parsed = await parse_upload(file, question)
    sid = session_id or str(uuid4())
    session = session_service.get_or_create(
        sid, system_prompt or FILE_ANALYSIS_SYSTEM_PROMPT
    )
    session_service.trim_messages(session)

    session["file"] = parsed.filename
    if parsed.df is not None:
        session["df"] = parsed.df
        session["df_summary"] = parsed.df_summary_text
    if parsed.file_raw is not None:
        session["file_raw"] = parsed.file_raw
        session["file_content_type"] = parsed.content_type

    session["messages"].append({"role": "user", "content": parsed.user_content})
    session["display"].append({"role": "user", "content": question})

    analysis_text = await claude_service.complete(
        session["messages"],
        system=session["system"],
        model=settings.claude_model,
        max_tokens=1200,
    )
    session["messages"].append({"role": "assistant", "content": analysis_text})
    session["display"].append({"role": "assistant", "content": analysis_text})

    artifacts = []
    forecast = None
    try:
        artifacts = await visualization_service.build_artifacts(sid, session, max_charts=3)
        forecast = await visualization_service.build_forecast(sid, session)
    except Exception:
        pass

    async def _bg_upload():
        resource_type = "image" if parsed.content_type.startswith("image/") else "raw"
        try:
            result = await storage_service.upload_file(
                parsed.raw,
                public_id=f"sessions/{sid}/{parsed.filename}",
                resource_type=resource_type,
            )
            session["cloudinary_id"] = result["public_id"]
            session["cloudinary_url"] = result["secure_url"]
        except Exception as exc:
            logger.error("Cloudinary upload failed: %s", exc)
        await storage_service.upload_state(sid, {
            "file": session.get("file"),
            "cloudinary_id": session.get("cloudinary_id"),
            "cloudinary_url": session.get("cloudinary_url"),
            "system": session.get("system"),
            "display": session.get("display", []),
        })

    asyncio.create_task(_bg_upload())
    session_service.touch(sid)

    from app.utils.html_utils import templates
    kpi_html = visualization_service.build_kpi_cards_html(session)
    cjs_html, cjs_init = visualization_service.build_chartjs_html(session)
    jsx_html = visualization_service.build_jsx_snippet(session, analysis_text) if session.get("df") is not None else ""

    return HTMLResponse(content=templates.get_template("dashboard.html").render({
        "request": None,
        "session_id": sid,
        "filename": parsed.filename,
        "question": question,
        "analysis_html": analysis_to_html(analysis_text),
        "kpi_html": kpi_html,
        "chartjs_html": cjs_html,
        "chart_init_js": cjs_init,
        "artifacts": artifacts,
        "forecast": forecast,
        "jsx_html": jsx_html,
    }))
