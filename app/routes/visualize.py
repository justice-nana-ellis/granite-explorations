import logging

import requests as http_requests
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import HTMLResponse

from app.services.session_service import session_service
from app.services.storage_service import storage_service
from app.services.visualization_service import chart_store, visualization_service
from app.utils.html_utils import analysis_to_html

logger = logging.getLogger(__name__)
router = APIRouter(tags=["visualize"])


@router.get("/charts/{session_id}/{filename}")
def serve_chart(session_id: str, filename: str):
    data = chart_store.get(f"{session_id}/{filename}")
    if not data:
        raise HTTPException(status_code=404, detail="Chart not found")
    return Response(content=data, media_type="image/png")


@router.get("/visualize/{session_id}")
async def visualize_session(session_id: str, chart_type: str = "auto", max_charts: int = 3):
    session = session_service.get(session_id)
    if session is None:
        session = session_service.get_or_create(session_id, "You are a helpful assistant.")

    artifacts = await visualization_service.build_artifacts(
        session_id, session, chart_type=chart_type, max_charts=max_charts
    )
    session_service.touch(session_id)
    return {"session_id": session_id, "count": len(artifacts), "artifacts": artifacts}


@router.get("/forecast/{session_id}")
async def forecast_session(session_id: str, periods: int = 3):
    session = session_service.get(session_id)
    if session is None:
        session = session_service.get_or_create(session_id, "You are a helpful assistant.")

    forecast = await visualization_service.build_forecast(session_id, session, periods=periods)
    session_service.touch(session_id)
    return {"session_id": session_id, "forecast": forecast}


@router.get("/artifacts/{session_id}")
async def list_session_artifacts(session_id: str):
    try:
        artifacts = await storage_service.list_session_artifacts(session_id)
        return {"session_id": session_id, "count": len(artifacts), "artifacts": artifacts}
    except Exception as exc:
        logger.exception("Cloudinary artifacts list failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Cloudinary error: {exc}")


@router.get("/dashboard/{session_id}/render")
async def render_dashboard(session_id: str):
    session = session_service.get(session_id)
    if session is None:
        session = session_service.get_or_create(session_id, "You are a helpful assistant.")

    artifacts = []
    forecast = None
    try:
        artifacts = await visualization_service.build_artifacts(session_id, session, max_charts=3)
        forecast = await visualization_service.build_forecast(session_id, session)
    except Exception:
        pass

    from app.utils.html_utils import templates
    kpi_html = visualization_service.build_kpi_cards_html(session)
    cjs_html, cjs_init = visualization_service.build_chartjs_html(session)
    jsx_html = visualization_service.build_jsx_snippet(session, "Dashboard") if session.get("df") is not None else ""

    session_service.touch(session_id)
    return HTMLResponse(content=templates.get_template("dashboard.html").render({
        "request": None,
        "session_id": session_id,
        "filename": session.get("file") or "Session Data",
        "question": "Open dashboard",
        "analysis_html": analysis_to_html("Dashboard generated from the uploaded session data."),
        "kpi_html": kpi_html,
        "chartjs_html": cjs_html,
        "chart_init_js": cjs_init,
        "artifacts": artifacts,
        "forecast": forecast,
        "jsx_html": jsx_html,
    }))


@router.get("/visualize/{session_id}/render")
async def render_first_artifact(session_id: str):
    session = session_service.get(session_id)
    if session is None:
        session = session_service.get_or_create(session_id, "You are a helpful assistant.")

    artifacts = await visualization_service.build_artifacts(session_id, session, max_charts=1)
    if not artifacts:
        raise HTTPException(status_code=404, detail="No artifacts generated for this session.")

    resp = http_requests.get(artifacts[0]["url"], timeout=30)
    resp.raise_for_status()
    return Response(content=resp.content, media_type="image/png")
