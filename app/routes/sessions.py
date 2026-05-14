import logging

from fastapi import APIRouter, HTTPException

from app.services.session_service import session_service
from app.services.storage_service import storage_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["sessions"])


@router.get("/")
def root():
    from app.config import settings
    return {"status": "ok", "model": settings.claude_model, "port": settings.port}


@router.get("/health")
def health():
    return {"status": "healthy"}


@router.get("/sessions")
def list_sessions():
    session_service.purge_expired()
    all_sessions = session_service.get_all()
    return {
        "count": len(all_sessions),
        "sessions": [
            {"session_id": sid, "file": s["file"], "messages": s["display"]}
            for sid, s in all_sessions.items()
        ],
    }


@router.delete("/session/{session_id}")
def clear_session(session_id: str):
    session_service.delete(session_id)
    return {"cleared": session_id}


@router.get("/files")
async def list_files():
    try:
        files = await storage_service.list_session_files()
        return {"count": len(files), "files": files}
    except Exception as exc:
        logger.exception("Cloudinary list_files error")
        raise HTTPException(status_code=500, detail=f"Cloudinary error: {exc}")
