"""
Abstract LLM Client Interface

Defines the provider-agnostic contract for LLM interactions.
No third-party imports — stdlib only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional, Union


@dataclass
class ImagePart:
    """An image to include in a multimodal prompt."""
    data: bytes
    mime_type: str = "image/png"


@dataclass
class GenerateConfig:
    """
    Generation parameters, independent of any specific LLM provider.

    Notes:
        - thinking_budget: None = don't configure thinking (provider default);
          0 = disabled; -1 = dynamic (provider chooses); positive int = explicit budget.
        - response_json_schema + response_mime_type: if schema is set the concrete
          client will force response_mime_type to "application/json" automatically.
    """
    temperature: float = 0.7
    top_p: float = 0.95
    max_output_tokens: int = 4096
    response_mime_type: Optional[str] = None
    response_json_schema: Optional[dict] = None
    thinking_budget: Optional[int] = None  # None = omit ThinkingConfig entirely
    system_instruction: Optional[str] = None


@dataclass
class LLMResponse:
    """Response returned by all LLMClient methods."""
    text: str
    model: str = ""
    usage_metadata: Optional[dict] = None


# Union of supported content part types
ContentPart = Union[str, ImagePart]


class LLMClient(ABC):
    """
    Abstract base for LLM provider clients.

    All methods accept `contents` as either a bare string (text-only) or a list
    of ContentPart (strings and/or ImageParts for multimodal prompts).

    The model is bound at construction time; it is not passed per-call.
    """

    @abstractmethod
    def generate(
        self,
        contents: Union[str, List[ContentPart]],
        config: Optional[GenerateConfig] = None,
    ) -> LLMResponse:
        """Synchronous, blocking generation."""
        ...

    @abstractmethod
    async def generate_async(
        self,
        contents: Union[str, List[ContentPart]],
        config: Optional[GenerateConfig] = None,
    ) -> LLMResponse:
        """Async generation, returns complete response when done."""
        ...

    @abstractmethod
    async def generate_stream(
        self,
        contents: Union[str, List[ContentPart]],
        config: Optional[GenerateConfig] = None,
    ) -> AsyncIterator[str]:
        """Async streaming generation; yields text chunks as they arrive."""
        ...

    async def aclose(self) -> None:
        """Release any async resources (connections, etc.). Override if needed."""
        pass
