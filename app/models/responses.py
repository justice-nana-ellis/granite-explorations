from typing import Optional, Any
from pydantic import BaseModel


class ChatResponse(BaseModel):
    reply: str
    model: str


class ArtifactResponse(BaseModel):
    title: str
    chart_type: str
    x: str
    y: str
    public_id: str
    url: str
    forecast_rows: list[dict] = []
    summary: Optional[str] = None


class SessionMeta(BaseModel):
    session_id: str
    file: Optional[str]
    messages: list[dict]


class SessionListResponse(BaseModel):
    count: int
    sessions: list[SessionMeta]


class UploadFileResponse(BaseModel):
    session_id: str
    file: str
    content_type: str
    size_bytes: int
    rows: Optional[int]
    columns: Optional[list[str]]
    summary_preview: str
    status: str
    next: str


class AnalysisBundle(BaseModel):
    session_id: str
    file: Optional[str]
    analysis: str
    visuals: Optional[Any] = None
