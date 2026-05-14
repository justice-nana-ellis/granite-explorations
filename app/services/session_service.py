"""In-memory session store with TTL expiry and Cloudinary state restore."""
from __future__ import annotations

import asyncio
import io
import logging
import threading
from time import time
from typing import Optional

import pandas as pd

from app.config import settings
from app.models.session import SessionData
from app.utils.df_utils import df_summary

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_sessions: dict[str, SessionData] = {}


class SessionService:
    def get_all(self) -> dict[str, SessionData]:
        self.purge_expired()
        with _lock:
            return dict(_sessions)

    def get(self, session_id: str) -> Optional[SessionData]:
        with _lock:
            session = _sessions.get(session_id)
            if session:
                session["last_accessed"] = time()
            return session

    def create(self, session_id: str, system: str) -> SessionData:
        with _lock:
            session: SessionData = {
                "system": system,
                "messages": [],
                "display": [],
                "file": None,
                "file_content_type": "",
                "file_raw": None,
                "file_summary": None,
                "cloudinary_id": None,
                "cloudinary_url": None,
                "state_cloudinary_id": None,
                "df": None,
                "df_summary": None,
                "last_accessed": time(),
            }
            _sessions[session_id] = session
        return session

    def get_or_create(self, session_id: str, system: str) -> SessionData:
        self.purge_expired()
        session = self.get(session_id)
        if session:
            return session

        with _lock:
            if len(_sessions) >= settings.session_max_count:
                oldest = min(_sessions, key=lambda k: _sessions[k]["last_accessed"])
                self._delete_cloudinary(_sessions[oldest])
                del _sessions[oldest]

        session = self.create(session_id, system)
        return session

    def delete(self, session_id: str) -> Optional[SessionData]:
        with _lock:
            session = _sessions.pop(session_id, None)
        if session:
            self._delete_cloudinary(session)
        return session

    def touch(self, session_id: str) -> None:
        with _lock:
            if session_id in _sessions:
                _sessions[session_id]["last_accessed"] = time()

    def purge_expired(self) -> None:
        now = time()
        with _lock:
            expired = [
                sid for sid, s in _sessions.items()
                if now - s["last_accessed"] > settings.session_ttl
            ]
        for sid in expired:
            with _lock:
                session = _sessions.pop(sid, None)
            if session:
                self._delete_cloudinary(session)

    def trim_messages(self, session: SessionData) -> None:
        max_msgs = settings.session_max_messages
        if len(session["messages"]) >= max_msgs:
            first = session["messages"][:1]
            rest = session["messages"][1:][-(max_msgs - 2):]
            session["messages"] = first + rest
        if len(session["display"]) >= max_msgs:
            session["display"] = session["display"][-max_msgs:]

    def meta(self, session_id: str) -> dict:
        s = _sessions[session_id]
        return {"session_id": session_id, "file": s["file"], "messages": s["display"]}

    async def restore_from_cloudinary(self, session_id: str, session: SessionData) -> None:
        from app.services.storage_service import storage_service

        state = await storage_service.fetch_state(session_id)
        if not state:
            return
        session["file"] = state.get("file")
        session["cloudinary_id"] = state.get("cloudinary_id")
        session["cloudinary_url"] = state.get("cloudinary_url")
        session["system"] = state.get("system") or session["system"]
        session["display"] = state.get("display", [])
        session["state_cloudinary_id"] = f"sessions/{session_id}/state.json"

    async def reload_df(self, session: SessionData) -> None:
        if not session.get("cloudinary_url") or not session.get("file"):
            return
        import requests as http_requests

        try:
            resp = await asyncio.to_thread(
                http_requests.get, session["cloudinary_url"], timeout=30
            )
            resp.raise_for_status()
            raw = resp.content
            filename: str = session["file"]  # type: ignore[assignment]
            if filename.endswith(".csv"):
                session["df"] = pd.read_csv(io.BytesIO(raw), low_memory=False)
            elif filename.endswith((".xlsx", ".xls")):
                session["df"] = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
            if session["df"] is not None:
                session["df_summary"] = df_summary(session["df"], filename)
        except Exception as exc:
            logger.debug("DF reload failed: %s", exc)

    def _delete_cloudinary(self, session: SessionData) -> None:
        from app.services.storage_service import storage_service

        if session.get("cloudinary_id"):
            filename = session.get("file", "") or ""
            rtype = (
                "image"
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))
                else "raw"
            )
            storage_service.delete(session["cloudinary_id"], resource_type=rtype)
        if session.get("state_cloudinary_id"):
            storage_service.delete(session["state_cloudinary_id"], resource_type="raw")


session_service = SessionService()
