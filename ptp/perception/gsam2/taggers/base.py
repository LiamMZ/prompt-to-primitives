"""Base class for object label taggers used by GSAM2ObjectTracker."""

import logging
from abc import ABC, abstractmethod

import numpy as np


class BaseTagger(ABC):
    logger: logging.Logger = logging.getLogger(__name__)

    """
    Abstract base for taggers that produce GroundingDINO prompt strings from RGB images.

    Subclasses must implement :meth:`tag`, which accepts an ``(H, W, 3)`` uint8 numpy
    array and an optional list of *required_tags* (object names that must appear in the
    output regardless of what the model sees), and returns a ``(prompt_str, raw_str)``
    tuple where:

    - ``prompt_str`` is a period-separated label string ready for GroundingDINO
      (e.g. ``"cup. bottle. keyboard."``).
    - ``raw_str`` is the unprocessed tag output from the underlying model, useful
      for logging and debugging.
    """

    @abstractmethod
    def tag(
        self,
        rgb_image: np.ndarray,
        required_tags: list[str] | None = None,
    ) -> tuple[str, str]:
        """
        Generate object labels for the given RGB image.

        Args:
            rgb_image: ``(H, W, 3)`` uint8 numpy array in RGB order.
            required_tags: Object names that must be included in the returned prompt
                regardless of what the model detects (e.g. goal objects from the task).

        Returns:
            ``(prompt_str, raw_str)`` — GroundingDINO-ready prompt and raw model output.
        """
        ...

    def __call__(
        self,
        rgb_image: np.ndarray,
        required_tags: list[str] | None = None,
    ) -> tuple[str, str]:
        return self.tag(rgb_image, required_tags=required_tags)
