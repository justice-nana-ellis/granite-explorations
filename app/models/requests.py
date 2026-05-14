from typing import Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None
    session_id: Optional[str] = None
    response_format: Optional[str] = None
