"""OpenAI vision-based object tagger."""

import io
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml
from PIL import Image

from .base import BaseTagger
from ptp.llm_interface.base import GenerateConfig, ImagePart, LLMClient

_DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[4] / "config" / "tagger_prompts.yaml"


class OpenAITagger(BaseTagger):
    """Tagger that uses an LLMClient to generate object labels from an RGB image.

    Produces a GroundingDINO-ready prompt string from the model's response.
    The system prompt is loaded from config/tagger_prompts.yaml so it can be
    edited without touching code.

    Args:
        llm_client: Any ptp.llm_interface.LLMClient implementation.
        prompts_config_path: Override path to the YAML prompt config.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompts_config_path: Optional[Path] = None,
    ) -> None:
        self._llm_client = llm_client
        self._prompts_path = Path(prompts_config_path or _DEFAULT_PROMPTS_PATH)
        self._prompts_cache: Optional[dict] = None
        self._prompts_mtime: float = 0.0

    def _load_prompts(self) -> dict:
        mtime = self._prompts_path.stat().st_mtime
        if self._prompts_cache is not None and mtime == self._prompts_mtime:
            return self._prompts_cache
        data = yaml.safe_load(self._prompts_path.read_text()) or {}
        self._prompts_cache = data
        self._prompts_mtime = mtime
        return data

    @staticmethod
    def _tags_to_prompt(tags: list[str]) -> str:
        seen: set[str] = set()
        unique = []
        for t in tags:
            t = t.strip().lower()
            if t and t not in seen:
                seen.add(t)
                unique.append(t)
        return " ".join(t + "." for t in unique)

    @staticmethod
    def _encode_image(rgb_image: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(rgb_image).save(buf, format="PNG")
        return buf.getvalue()

    def tag(
        self,
        rgb_image: np.ndarray,
        required_tags: list[str] | None = None,
        task: str | None = None,
    ) -> Tuple[str, str]:
        """Run the LLM on *rgb_image* and return ``(prompt_str, raw_str)``.

        Args:
            rgb_image: ``(H, W, 3)`` uint8 numpy array in RGB order.
            required_tags: Object names that must appear in the output (e.g. goal
                objects from the current task).
            task: Natural language task description. When provided, switches to the
                task-focused prompt that limits detection to task-relevant objects
                and their blockers rather than listing everything in the scene.

        Returns:
            ``(prompt_str, raw_str)`` where *prompt_str* is a period-separated label
            string suitable for GroundingDINO and *raw_str* is the raw JSON response.
        """
        prompts = self._load_prompts()
        if task:
            template = prompts.get("task_focused_prompt", "").strip()
            system_prompt = template.replace("{task}", task)
        else:
            system_prompt = prompts.get("system_prompt", "").strip()

        img_bytes = self._encode_image(rgb_image)
        config = GenerateConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            max_output_tokens=256,
            response_mime_type="application/json",
        )
        response = self._llm_client.generate(
            [ImagePart(data=img_bytes, mime_type="image/png")],
            config=config,
        )
        raw = response.text or ""

        try:
            parsed = json.loads(raw)
            tags = parsed.get("tags", [])
            self.logger.info("OpenAITagger tags: %s", tags)
        except json.JSONDecodeError:
            self.logger.warning("OpenAITagger: failed to parse JSON response: %r", raw)
            tags = []

        return self._tags_to_prompt(tags), raw
