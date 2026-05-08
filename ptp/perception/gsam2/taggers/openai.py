"""OpenAI vision-based object tagger."""

import base64
import io
import json
import numpy as np
from PIL import Image

from .base import BaseTagger

_DEFAULT_MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT_BASE = (
    "You are a robotics perception system. "
    "Given an image, list every distinct object you can see that a robot could interact with. "
    "Rules:\n"
    "- Prefer a single generic noun per entry (e.g. 'cube', 'mug', 'bottle', 'table').\n"
    "- If multiple instances of the same category are visually distinct (e.g. different colors or "
    "materials), give each its own entry with a qualifying adjective "
    "(e.g. 'red block', 'blue block' rather than just 'block').\n"
    "- Do not emit duplicate entries — every tag in the list must be unique.\n"
    "- Omit vague words like 'object', 'item', 'thing', 'surface', 'background'.\n"
    'Respond with a JSON object in this exact format: {"tags": ["tag1", "tag2"]}'
)


class OpenAITagger(BaseTagger):
    """
    Tagger that uses an OpenAI vision model to generate object labels from an RGB image.

    Produces a GroundingDINO-ready prompt string from the model's response.

    Args:
        api_key: OpenAI API key. If omitted, the SDK reads ``OPENAI_API_KEY`` from env.
        model: OpenAI model ID (default: ``"gpt-4o-mini"``).
        max_tokens: Maximum tokens in the model response (default: 256).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = 256,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required. Install with: pip install openai"
            ) from e

        self._model = model
        self._max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key)

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
    def _encode_image(rgb_image: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(rgb_image).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def tag(
        self,
        rgb_image: np.ndarray,
        required_tags: list[str] | None = None,
    ) -> tuple[str, str]:
        """
        Run the OpenAI vision model on *rgb_image* and return ``(prompt_str, raw_str)``.

        Args:
            rgb_image: ``(H, W, 3)`` uint8 numpy array in RGB order.
            required_tags: Object names that must appear in the output (e.g. goal objects
                from the current task). These are appended to the system prompt so the
                model is explicitly told to look for them.

        Returns:
            ``(prompt_str, raw_str)`` where *prompt_str* is a period-separated label
            string suitable for GroundingDINO and *raw_str* is the raw JSON response
            text from the model.
        """
        b64 = self._encode_image(rgb_image)

        system_prompt = _SYSTEM_PROMPT_BASE
        if required_tags:
            required_str = ", ".join(f"'{t}'" for t in required_tags)
            system_prompt += (
                f"\nAdditionally, you should look for the following task-relevant objects "
                f"in your response if they are present in the image: {required_str}."
            )

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        }
                    ],
                },
            ],
        )

        raw = response.choices[0].message.content or ""

        try:
            parsed = json.loads(raw)
            tags = parsed.get("tags", [])
            self.logger.info(f"OpenAI tags: {tags}")
        except json.JSONDecodeError:
            self.logger.warning("OpenAITagger: failed to parse JSON response: %r", raw)
            tags = []

        return self._tags_to_prompt(tags), raw
