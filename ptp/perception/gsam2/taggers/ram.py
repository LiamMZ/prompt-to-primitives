"""RAM+ tagger implementation."""

import numpy as np
from PIL import Image

from .base import BaseTagger


class RAMTagger(BaseTagger):
    """
    Tagger backed by the RAM+ (Recognize Anything Model Plus) image tagging model.

    Loads a ``ram_plus`` checkpoint and produces a GroundingDINO-ready prompt string
    from the pipe-separated tag output that RAM+ generates.

    Args:
        ckpt_path: Path to the RAM+ ``.pth`` checkpoint file.
        image_size: Input resolution expected by the model (default: 384).
        device: Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
    """

    def __init__(self, ckpt_path: str, image_size: int = 384, device: str = "cuda"):
        import torch
        from ram.models import ram_plus
        from ram import get_transform

        self._device = device
        self._transform = get_transform(image_size=image_size)

        self._model = ram_plus(pretrained=ckpt_path, image_size=image_size, vit="swin_l")
        self._model.eval().to(device)

    @staticmethod
    def _tags_to_prompt(tags: list[str]) -> str:
        clean = [t.strip().lower() for t in tags if t.strip()]
        return " ".join(t + "." for t in clean)

    def tag(self, rgb_image: np.ndarray, required_tags: list[str] | None = None) -> tuple[str, str]:
        """
        Run RAM+ on *rgb_image* and return ``(prompt_str, raw_str)``.

        Args:
            rgb_image: ``(H, W, 3)`` uint8 numpy array in RGB order.

        Returns:
            ``(prompt_str, raw_str)`` where *prompt_str* is a period-separated label
            string suitable for GroundingDINO and *raw_str* is the raw pipe-separated
            tag output from RAM+.
        """
        import torch
        pil_img = Image.fromarray(rgb_image)
        tensor = self._transform(pil_img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            tags, _ = self._model.generate_tag(tensor)
            print(f"RAM+ tags: {tags}")
        raw = tags[0] if isinstance(tags, (list, tuple)) else tags
        tag_list = [t.strip() for t in raw.split("|") if t.strip()]
        return self._tags_to_prompt(tag_list), raw
