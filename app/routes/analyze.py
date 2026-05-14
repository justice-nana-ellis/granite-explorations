import base64
import logging
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app.config import settings
from app.models.requests import ChatRequest
from app.services.claude_service import claude_service
from app.services.session_service import session_service
from app.services.visualization_service import visualization_service
from app.utils.html_utils import analysis_to_html
from app.utils.prompt_library import ANALYSIS_SYSTEM_PROMPT

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analyze"])

IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})


@router.post("/analyze")
async def analyze(request: ChatRequest, http_request: Request):
    sid = request.session_id or str(uuid4())
    session = session_service.get_or_create(sid, request.system_prompt or ANALYSIS_SYSTEM_PROMPT)
    session_service.trim_messages(session)

    if session.get("df") is None and session.get("cloudinary_url"):
        await session_service.reload_df(session)

    ct = session.get("file_content_type", "")
    if session.get("df_summary"):
        msg = f"[File context]\n{session['df_summary']}\n\n[Question]\n{request.message}"
    elif ct in IMAGE_TYPES and session.get("file_raw"):
        encoded = base64.standard_b64encode(session["file_raw"]).decode("utf-8")
        msg = [
            {"type": "image", "source": {"type": "base64", "media_type": ct, "data": encoded}},
            {"type": "text", "text": request.message},
        ]
    elif ct == "application/pdf" and session.get("file_raw"):
        encoded = base64.standard_b64encode(session["file_raw"]).decode("utf-8")
        msg = [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": encoded}},
            {"type": "text", "text": request.message},
        ]
    elif session.get("file_summary"):
        msg = f"[File context]\n{session['file_summary']}\n\n[Question]\n{request.message}"
    else:
        msg = request.message

    session["messages"].append({"role": "user", "content": msg})
    session["display"].append({"role": "user", "content": request.message})

    base_url = str(http_request.base_url).rstrip("/")
    wants_json = (request.response_format or "").lower() in {"json", "bundle"}

    analysis_text = await claude_service.complete(
        session["messages"],
        system=session["system"],
        model=settings.claude_chat_model,
        max_tokens=4096,
    )
    session["messages"].append({"role": "assistant", "content": analysis_text})
    session["display"].append({"role": "assistant", "content": analysis_text})

    artifacts: list = []
    forecast = None
    visual_bundle: dict = {}
    if session.get("df") is not None:
        try:
            visual_bundle = await visualization_service.build_visual_bundle(sid, session, base_url)
            artifacts = visual_bundle.get("artifacts", [])
            forecast = visual_bundle.get("forecast")
        except Exception:
            logger.exception("Visual bundle build failed for session %s", sid)

    session_service.touch(sid)

    if wants_json:
        return {
            "session_id": sid,
            "file": session.get("file"),
            "analysis": analysis_text,
            "visuals": visual_bundle or None,
        }

    from app.utils.html_utils import templates
    kpi_html = visualization_service.build_kpi_cards_html(session)
    cjs_html, cjs_init = visualization_service.build_chartjs_html(session)
    jsx_html = visualization_service.build_jsx_snippet(session, analysis_text) if session.get("df") is not None else ""

    return HTMLResponse(content=templates.get_template("dashboard.html").render({
        "request": http_request,
        "session_id": sid,
        "filename": session.get("file") or "Session Chat",
        "question": request.message,
        "analysis_html": analysis_to_html(analysis_text),
        "kpi_html": kpi_html,
        "chartjs_html": cjs_html,
        "chart_init_js": cjs_init,
        "artifacts": artifacts,
        "forecast": forecast,
        "jsx_html": jsx_html,
    }))
