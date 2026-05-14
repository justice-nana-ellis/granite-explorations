"""Anthropic AsyncAnthropic client wrapper."""
from __future__ import annotations

import logging
from typing import AsyncIterator

from anthropic import AsyncAnthropic

from app.config import settings

logger = logging.getLogger(__name__)

_client: AsyncAnthropic | None = None


def get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.claude_api_key)
    return _client


class ClaudeService:
    async def stream(
        self,
        messages: list[dict],
        system: str,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        use_model = model or settings.claude_chat_model
        async with get_client().messages.stream(
            model=use_model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def complete(
        self,
        messages: list[dict],
        system: str,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> str:
        use_model = model or settings.claude_chat_model
        kwargs = dict(
            model=use_model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = await get_client().messages.create(**kwargs)
        return _extract_text(response)


def _extract_text(response) -> str:
    text_parts = [
        block.text
        for block in getattr(response, "content", []) or []
        if hasattr(block, "text") and block.text
    ]
    return "\n".join(text_parts).strip() or "No assistant response generated."


claude_service = ClaudeService()
