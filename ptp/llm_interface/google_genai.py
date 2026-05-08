"""
Google GenAI LLM Client

Concrete LLMClient implementation wrapping google.genai.Client.
All provider-specific knowledge (types, API calls, response shapes) is
contained here — callers only need the base interface types.

Usage:
    from src.llm_interface import GoogleGenAIClient, GenerateConfig, ImagePart

    llm = GoogleGenAIClient(model="gemini-2.0-flash", api_key="...")

    # Text only
    response = llm.generate("What is 2+2?")

    # Multimodal
    with open("photo.png", "rb") as f:
        img = ImagePart(data=f.read())
    response = llm.generate([img, "Describe this image."])

    # Structured JSON
    cfg = GenerateConfig(response_mime_type="application/json", temperature=0.1)
    response = llm.generate("Return a JSON object with key 'answer'.", config=cfg)

    # Async streaming
    async for chunk in llm.generate_stream("Tell me a story."):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, List, Optional, Union

from ptp.llm_interface.base import (
    ContentPart,
    GenerateConfig,
    ImagePart,
    LLMClient,
    LLMResponse,
)

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    genai = None  # type: ignore
    genai_types = None  # type: ignore

logger = logging.getLogger(__name__)


class GoogleGenAIClient(LLMClient):
    """
    LLMClient backed by google.genai.Client.

    Args:
        model:   Model identifier string (e.g. "gemini-2.0-flash").
        api_key: Google AI API key. If omitted, the SDK uses GOOGLE_API_KEY env var.
        client:  Optional pre-built genai.Client to reuse (e.g. one already wrapped
                 by genai_logging). When provided, api_key is ignored.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,  # genai.Client
    ) -> None:
        if not _GENAI_AVAILABLE:
            raise ImportError(
                "google-genai is required. Install with: pip install google-genai"
            )
        self._model = model
        if client is not None:
            self._client = client
        else:
            self._client = genai.Client(api_key=api_key)

    # ------------------------------------------------------------------
    # LLMClient implementation
    # ------------------------------------------------------------------

    def generate(
        self,
        contents: Union[str, List[ContentPart]],
        config: Optional[GenerateConfig] = None,
    ) -> LLMResponse:
        sdk_contents = self._translate_contents(contents)
        sdk_config = self._build_config(config)
        response = self._client.models.generate_content(
            model=self._model,
            contents=sdk_contents,
            config=sdk_config,
        )
        return LLMResponse(
            text=response.text or "",
            model=self._model,
            usage_metadata=self._extract_usage(response),
        )

    async def generate_async(
        self,
        contents: Union[str, List[ContentPart]],
        config: Optional[GenerateConfig] = None,
    ) -> LLMResponse:
        sdk_contents = self._translate_contents(contents)
        sdk_config = self._build_config(config)
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=sdk_contents,
            config=sdk_config,
        )
        # Log finish reason and token usage to help diagnose truncation
        try:
            candidate = response.candidates[0] if response.candidates else None
            if candidate is not None:
                finish_reason = getattr(candidate, "finish_reason", None)
                usage = self._extract_usage(response)
                if finish_reason and str(finish_reason) not in ("FinishReason.STOP", "STOP", "1"):
                    logger.warning(
                        "generate_async finish_reason=%s tokens=%s — response may be truncated",
                        finish_reason, usage,
                    )
        except Exception:
            pass
        try:
            text = response.text
        except Exception:
            text = ""
            for candidate in (response.candidates or []):
                for part in (getattr(candidate.content, "parts", None) or []):
                    text += getattr(part, "text", "") or ""
        return LLMResponse(
            text=text or "",
            model=self._model,
            usage_metadata=self._extract_usage(response),
        )

    async def generate_stream(
        self,
        contents: Union[str, List[ContentPart]],
        config: Optional[GenerateConfig] = None,
    ) -> AsyncIterator[str]:
        sdk_contents = self._translate_contents(contents)
        sdk_config = self._build_config(config)
        stream = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=sdk_contents,
            config=sdk_config,
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    async def aclose(self) -> None:
        try:
            await self._client.aio.aclose()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _translate_contents(
        self, contents: Union[str, List[ContentPart]]
    ) -> List[Any]:
        """Convert abstract ContentPart list to google-genai SDK content list."""
        if isinstance(contents, str):
            return [contents]
        result = []
        for part in contents:
            if isinstance(part, ImagePart):
                result.append(
                    genai_types.Part.from_bytes(
                        data=part.data, mime_type=part.mime_type
                    )
                )
            else:
                result.append(part)
        return result

    def _build_config(
        self, config: Optional[GenerateConfig]
    ) -> "genai_types.GenerateContentConfig":
        """Translate GenerateConfig to google-genai GenerateContentConfig."""
        if config is None:
            return genai_types.GenerateContentConfig()

        kwargs: dict = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_output_tokens": config.max_output_tokens,
        }

        # Structured output — force mime type when schema is present
        mime = config.response_mime_type
        if config.response_json_schema is not None:
            mime = "application/json"
            kwargs["response_json_schema"] = config.response_json_schema
        if mime is not None:
            kwargs["response_mime_type"] = mime

        # Thinking config — only set when explicitly requested
        if config.thinking_budget is not None:
            kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=config.thinking_budget
            )

        return genai_types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _extract_usage(response: Any) -> Optional[dict]:
        """Extract token usage metadata if present, None otherwise."""
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return None
        try:
            return {
                "prompt_token_count": getattr(meta, "prompt_token_count", None),
                "candidates_token_count": getattr(meta, "candidates_token_count", None),
                "total_token_count": getattr(meta, "total_token_count", None),
            }
        except Exception:
            return None
