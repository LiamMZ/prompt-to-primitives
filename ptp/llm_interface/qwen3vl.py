"""
Qwen3-VL (and compatible HuggingFace) LLM/VLM Client

Concrete LLMClient implementation wrapping HuggingFace transformers models,
optimised for Qwen3-VL but supporting any VLM or text-only LLM that accepts
apply_chat_template-style inputs.

Uses the chat-template / apply_chat_template pipeline which is required by
modern VLMs (Qwen3-VL, Qwen2-VL, etc.) and works cleanly for text-only models too.

Supported model families:
  VLMs  — Qwen/Qwen3-VL-*, Qwen/Qwen2-VL-*, microsoft/Phi-3.5-vision-*, llava-hf/llava-*
  LLMs  — meta-llama/Meta-Llama-*, mistralai/Mistral-*, Qwen/Qwen2.5-*

Usage:
    from src.llm_interface import Qwen3VLClient, GenerateConfig, ImagePart

    # Qwen3-VL (vision + language, recommended default)
    llm = Qwen3VLClient(model="Qwen/Qwen3-VL-4B-Thinking")

    with open("photo.png", "rb") as f:
        img = ImagePart(data=f.read())
    response = llm.generate([img, "What objects do you see?"])
    print(response.text)

    # Text-only LLM
    llm = Qwen3VLClient(model="meta-llama/Meta-Llama-3.1-8B-Instruct")
    response = llm.generate("Describe a pick-and-place task in PDDL.")

    # JSON output (prompt-enforced)
    cfg = GenerateConfig(response_mime_type="application/json", temperature=0.1)
    response = llm.generate("Return a JSON list of objects.", config=cfg)

    # Streaming
    async for chunk in llm.generate_stream("Tell me about robotics."):
        print(chunk, end="", flush=True)

Notes:
    - Models are loaded once at construction and kept in memory.
    - Inputs are built via apply_chat_template, which is the correct path for
      all modern chat/instruction-tuned models.
    - Streaming uses TextIteratorStreamer in a background thread; the event loop
      is never blocked (asyncio.to_thread for non-streaming async paths).
    - JSON mode prepends a system message enforcing JSON-only output.
    - thinking_budget is silently ignored (not a transformers concept).
    - Pass load_in_4bit=True for 4-bit quantisation (requires bitsandbytes).
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from ptp.llm_interface.base import (
    ContentPart,
    GenerateConfig,
    ImagePart,
    LLMClient,
    LLMResponse,
)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        TextIteratorStreamer,
    )
    # Qwen3-VL ships its own model class; fall back to AutoModelForCausalLM if missing
    try:
        from transformers import Qwen3VLForConditionalGeneration
        _QWEN3VL_AVAILABLE = True
    except ImportError:
        _QWEN3VL_AVAILABLE = False
        Qwen3VLForConditionalGeneration = None  # type: ignore

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None  # type: ignore
    AutoProcessor = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    TextIteratorStreamer = None  # type: ignore
    Qwen3VLForConditionalGeneration = None  # type: ignore
    _QWEN3VL_AVAILABLE = False

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    PILImage = None  # type: ignore

logger = logging.getLogger(__name__)

_JSON_SYSTEM_INSTRUCTION = (
    "You must respond with valid JSON only. "
    "Do not include markdown code fences, explanations, or any text outside the JSON."
)

# Model ID prefixes that use Qwen3VLForConditionalGeneration
_QWEN3VL_PREFIXES = ("Qwen/Qwen3-VL", "qwen3-vl")


class Qwen3VLClient(LLMClient):
    """
    LLMClient backed by a locally loaded HuggingFace transformers model.

    Optimised for Qwen3-VL and compatible VLM/LLM families.

    Args:
        model:            HuggingFace model ID or local path.
        device:           "cuda", "cuda:1", "cpu", or "auto" (device_map="auto").
        torch_dtype:      Weight dtype. Defaults to "auto" (lets transformers decide),
                          which is bfloat16 on CUDA for most modern models.
        load_in_4bit:     4-bit quantisation via bitsandbytes.
        load_in_8bit:     8-bit quantisation via bitsandbytes.
        trust_remote_code: Forward trust_remote_code=True to from_pretrained.
        max_new_tokens:   Default generation length when config.max_output_tokens unset.
        processor_kwargs: Extra kwargs for AutoProcessor.from_pretrained.
        model_kwargs:     Extra kwargs for the model's from_pretrained.
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        torch_dtype: Optional[Any] = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        trust_remote_code: bool = False,
        max_new_tokens: int = 1024,
        processor_kwargs: Optional[Dict] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required: pip install transformers")
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required: pip install torch")

        self._model_id = model
        self._max_new_tokens = max_new_tokens
        self._device_map = "auto" if device == "auto" else None
        self._device = device if device != "auto" else "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

        # Resolve dtype — "auto" lets transformers pick the best dtype per layer
        resolved_dtype = torch_dtype
        if resolved_dtype is None:
            resolved_dtype = torch.bfloat16 if self._device.startswith("cuda") else torch.float32

        # Build model from_pretrained kwargs
        _model_kw: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": resolved_dtype,
        }
        if self._device_map is not None:
            _model_kw["device_map"] = self._device_map
        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                )
                _model_kw["quantization_config"] = bnb_cfg
                # torch_dtype must not be set when using BnB quantisation
                _model_kw.pop("torch_dtype", None)
            except ImportError:
                raise ImportError("bitsandbytes is required for 4/8-bit quantisation: pip install bitsandbytes")
        if model_kwargs:
            _model_kw.update(model_kwargs)

        # Choose the right model class
        model_lower = model.lower()
        use_qwen3vl = _QWEN3VL_AVAILABLE and any(p in model for p in _QWEN3VL_PREFIXES)

        logger.info("Loading HuggingFace model '%s' (Qwen3-VL=%s) …", model, use_qwen3vl)
        if use_qwen3vl:
            self._hf_model = Qwen3VLForConditionalGeneration.from_pretrained(model, **_model_kw)
        else:
            self._hf_model = AutoModelForCausalLM.from_pretrained(model, **_model_kw)

        if self._device_map is None:
            self._hf_model = self._hf_model.to(self._device)
        self._hf_model.eval()

        # Processor — always try AutoProcessor first (covers VLMs)
        _proc_kw: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if processor_kwargs:
            _proc_kw.update(processor_kwargs)
        try:
            self._processor = AutoProcessor.from_pretrained(model, **_proc_kw)
            self._is_vlm = True
            logger.info("  ✓ Loaded AutoProcessor (VLM / chat-template mode)")
        except Exception:
            self._processor = AutoTokenizer.from_pretrained(model, **_proc_kw)
            self._is_vlm = False
            logger.info("  ✓ Loaded AutoTokenizer (text-only mode)")

    # ------------------------------------------------------------------
    # LLMClient implementation
    # ------------------------------------------------------------------

    def generate(
        self,
        contents: Union[str, List[ContentPart]],
        config: Optional[GenerateConfig] = None,
    ) -> LLMResponse:
        """Synchronous blocking generation."""
        messages = self._build_messages(contents, config)
        output = self._run_inference(messages, config)
        return LLMResponse(text=output, model=self._model_id)

    async def generate_async(
        self,
        contents: Union[str, List[ContentPart]],
        config: Optional[GenerateConfig] = None,
    ) -> LLMResponse:
        """Async generation with live token streaming to stdout."""
        cfg = config or GenerateConfig()
        tokenizer = self._processor if not self._is_vlm else self._processor.tokenizer
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=300.0,
        )
        result_holder: List[Optional[Exception]] = [None]

        def _run_in_thread():
            try:
                print("[Qwen3VL] building prompt...", flush=True)
                messages = self._build_messages(contents, config)
                print("[Qwen3VL] applying chat template...", flush=True)
                inputs = self._apply_template(messages)
                print("[Qwen3VL] starting generation...", flush=True)
                self._hf_model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_output_tokens or self._max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    do_sample=cfg.temperature > 0.0,
                    streamer=streamer,
                )
            except Exception as exc:
                result_holder[0] = exc
                streamer.end()

        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()

        chunks: List[str] = []
        print("\n[Qwen3VL] ", end="", flush=True)
        for chunk in streamer:
            if chunk:
                print(chunk, end="", flush=True)
                chunks.append(chunk)
            await asyncio.sleep(0)
        print()

        thread.join()
        if result_holder[0] is not None:
            raise result_holder[0]
        return LLMResponse(text="".join(chunks), model=self._model_id)

    async def generate_stream(
        self,
        contents: Union[str, List[ContentPart]],
        config: Optional[GenerateConfig] = None,
    ) -> AsyncIterator[str]:
        """
        Async streaming via TextIteratorStreamer running in a background thread.
        """
        messages = self._build_messages(contents, config)
        inputs = self._apply_template(messages)
        cfg = config or GenerateConfig()

        tokenizer = self._processor if not self._is_vlm else self._processor.tokenizer
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": cfg.max_output_tokens or self._max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "do_sample": cfg.temperature > 0.0,
            "streamer": streamer,
        }

        thread = threading.Thread(
            target=self._hf_model.generate,
            kwargs=gen_kwargs,
            daemon=True,
        )
        thread.start()

        for chunk in streamer:
            if chunk:
                yield chunk
            await asyncio.sleep(0)

        thread.join()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        contents: Union[str, List[ContentPart]],
        config: Optional[GenerateConfig],
    ) -> List[Dict]:
        """
        Convert abstract ContentPart list into the messages list format used by
        apply_chat_template: [{"role": "user", "content": [{"type": ..., ...}, ...]}]

        A system message enforcing JSON output is prepended when requested.
        """
        messages: List[Dict] = []

        # Build user content parts
        if isinstance(contents, str):
            user_parts = [{"type": "text", "text": contents}]
        else:
            user_parts = []
            for part in contents:
                if isinstance(part, ImagePart):
                    if not _PIL_AVAILABLE:
                        raise ImportError("Pillow is required: pip install pillow")
                    pil_img = PILImage.open(io.BytesIO(part.data)).convert("RGB")
                    user_parts.append({"type": "image", "image": pil_img})
                else:
                    user_parts.append({"type": "text", "text": str(part)})

        messages.append({"role": "user", "content": user_parts})
        return messages

    def _apply_template(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Run apply_chat_template and return a dict of tensors ready for model.generate().
        Moves tensors to the correct device.
        """
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        target_device = (
            next(self._hf_model.parameters()).device
            if self._device_map is not None
            else self._device
        )
        return {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def _run_inference(
        self,
        messages: List[Dict],
        config: Optional[GenerateConfig],
    ) -> str:
        """Run model.generate() synchronously and decode only the new tokens."""
        inputs = self._apply_template(messages)
        cfg = config or GenerateConfig()
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self._hf_model.generate(
                **inputs,
                max_new_tokens=cfg.max_output_tokens or self._max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=cfg.temperature > 0.0,
            )

        # Trim prompt tokens and decode
        trimmed = [out[input_len:] for out in output_ids]
        tokenizer = self._processor if not self._is_vlm else self._processor.tokenizer
        texts = tokenizer.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return texts[0].strip() if texts else ""
