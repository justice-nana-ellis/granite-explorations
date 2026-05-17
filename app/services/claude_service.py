"""Anthropic AsyncAnthropic client wrapper."""
from __future__ import annotations

import logging
from typing import AsyncIterator

import httpx
from anthropic import AsyncAnthropic

from app.config import settings

logger = logging.getLogger(__name__)

_client: AsyncAnthropic | None = None


def get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        # connect=10s, read=600s — large forecast responses can take several minutes
        _client = AsyncAnthropic(
            api_key=settings.claude_api_key,
            timeout=httpx.Timeout(600.0, connect=10.0),
        )
    return _client


class ClaudeService:
    async def stream(
        self,
        messages: list[dict],
        system: str,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        use_model = model or settings.claude_chat_model
        kwargs = dict(
            model=use_model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        async with get_client().messages.stream(**kwargs) as stream:
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

    async def complete_with_tool(
        self,
        messages: list[dict],
        system: str,
        tool: dict,
        model: str | None = None,
        max_tokens: int = 16384,
    ) -> dict:
        """Call Claude with a forced tool choice and return the tool input as a dict."""
        use_model = model or settings.claude_model
        response = await get_client().messages.create(
            model=use_model,
            max_tokens=max_tokens,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=messages,
        )
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise ValueError(
                f"Model hit max_tokens ({max_tokens}) before finishing — "
                "increase max_tokens or reduce context."
            )
        for block in response.content:
            if hasattr(block, "input") and isinstance(block.input, dict):
                if not block.input:
                    raise ValueError("Model returned an empty tool input.")
                return block.input
        raise ValueError("Model did not return a tool_use block.")


def _extract_text(response) -> str:
    text_parts = [
        block.text
        for block in getattr(response, "content", []) or []
        if hasattr(block, "text") and block.text
    ]
    return "\n".join(text_parts).strip() or "No assistant response generated."


claude_service = ClaudeService()
