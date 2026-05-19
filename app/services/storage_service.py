"""Cloudinary storage abstraction."""
from __future__ import annotations

import asyncio
import io
import json
import logging

import cloudinary
import cloudinary.api
import cloudinary.uploader

from app.config import settings

logger = logging.getLogger(__name__)


def configure_cloudinary() -> None:
    cloudinary.config(
        cloud_name=settings.cloudinary_cloud_name,
        api_key=settings.cloudinary_api_key,
        api_secret=settings.cloudinary_api_secret,
        secure=True,
    )


class StorageService:
    async def upload_file(
        self,
        data: bytes,
        public_id: str,
        resource_type: str = "raw",
    ) -> dict:
        try:
            return await asyncio.to_thread(
                cloudinary.uploader.upload,
                io.BytesIO(data),
                public_id=public_id,
                resource_type=resource_type,
                overwrite=True,
            )
        except Exception as exc:
            logger.error("Cloudinary upload failed for %s: %s", public_id, exc)
            raise

    async def upload_state(self, session_id: str, state: dict) -> str:
        raw_json = json.dumps(state).encode()
        result = await self.upload_file(
            raw_json,
            public_id=f"sessions/{session_id}/state.json",
            resource_type="raw",
        )
        return result["public_id"]

    async def fetch_state(self, session_id: str) -> dict | None:
        cloud_name = settings.cloudinary_cloud_name
        import requests as http_requests

        url = f"https://res.cloudinary.com/{cloud_name}/raw/upload/sessions/{session_id}/state.json"
        try:
            resp = await asyncio.to_thread(http_requests.get, url, timeout=10)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception as exc:
            logger.debug("State restore failed for session %s: %s", session_id, exc)
            return None

    async def list_rag_session_ids(self) -> list[str]:
        """Return all session_ids that have a state.json saved in Cloudinary."""
        def _fetch():
            results, next_cursor = [], None
            while True:
                kwargs: dict = {
                    "resource_type": "raw",
                    "type": "upload",
                    "prefix": "sessions/",
                    "max_results": 500,
                }
                if next_cursor:
                    kwargs["next_cursor"] = next_cursor
                resp = cloudinary.api.resources(**kwargs)
                results.extend(resp.get("resources", []))
                next_cursor = resp.get("next_cursor")
                if not next_cursor:
                    break
            return results

        all_resources = await asyncio.to_thread(_fetch)
        ids = []
        for r in all_resources:
            pub = r["public_id"]  # e.g. "sessions/<sid>/state.json"
            if pub.endswith("/state.json"):
                parts = pub.split("/")
                if len(parts) == 3:
                    ids.append(parts[1])
        return ids

    def delete(self, public_id: str, resource_type: str = "raw") -> None:
        try:
            cloudinary.uploader.destroy(public_id, resource_type=resource_type)
        except Exception as exc:
            logger.debug("Cloudinary delete failed for %s: %s", public_id, exc)

    async def list_session_files(self) -> list[dict]:
        def _fetch_all():
            results = []
            for rtype in ("image", "raw"):
                next_cursor = None
                while True:
                    kwargs: dict = dict(
                        resource_type=rtype,
                        type="upload",
                        prefix="sessions/",
                        max_results=500,
                    )
                    if next_cursor:
                        kwargs["next_cursor"] = next_cursor
                    resp = cloudinary.api.resources(**kwargs)
                    results.extend(resp.get("resources", []))
                    next_cursor = resp.get("next_cursor")
                    if not next_cursor:
                        break
            return results

        import os
        all_resources = await asyncio.to_thread(_fetch_all)
        return [
            {
                "public_id": r["public_id"],
                "url": r["secure_url"],
                "bytes": r["bytes"],
                "format": r.get("format") or os.path.splitext(r["public_id"])[-1].lstrip(".") or "unknown",
                "resource_type": r.get("resource_type"),
                "created_at": r.get("created_at"),
            }
            for r in all_resources
            if not r["public_id"].endswith("state.json")
        ]

    async def list_session_artifacts(self, session_id: str) -> list[dict]:
        import os

        def _fetch():
            resources = []
            prefix = f"sessions/{session_id}/artifacts/"
            next_cursor = None
            while True:
                kwargs: dict = {
                    "resource_type": "image",
                    "type": "upload",
                    "prefix": prefix,
                    "max_results": 500,
                }
                if next_cursor:
                    kwargs["next_cursor"] = next_cursor
                resp = cloudinary.api.resources(**kwargs)
                resources.extend(resp.get("resources", []))
                next_cursor = resp.get("next_cursor")
                if not next_cursor:
                    break
            return resources

        resources = await asyncio.to_thread(_fetch)
        return [
            {
                "public_id": r["public_id"],
                "url": r["secure_url"],
                "bytes": r.get("bytes"),
                "format": r.get("format") or os.path.splitext(r["public_id"])[-1].lstrip(".") or "unknown",
                "created_at": r.get("created_at"),
            }
            for r in resources
        ]


storage_service = StorageService()
