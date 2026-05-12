"""
Molmo pointing server.

Loads Molmo2-4B once and serves a /point HTTP endpoint so the main PTP
pipeline can query it without reloading the model on every run.

Usage:
    python scripts/molmo_server.py [--port 8765] [--checkpoint allenai/Molmo2-4B]

Endpoint:
    POST /point
    Content-Type: application/json
    Body: {"prompt": "Point to ...", "image_b64": "<base64-encoded PNG>"}
    Response: {"text": "<molmo raw output>", "image_w": <int>, "image_h": <int>}

    POST /health
    Response: {"status": "ok"}
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("molmo_server")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Molmo pointing inference server")
    p.add_argument("--port", type=int, default=8765, help="TCP port to listen on")
    p.add_argument(
        "--checkpoint",
        default=os.environ.get("MOLMO_CHECKPOINT", "allenai/Molmo2-4B"),
        help="HuggingFace model ID or local path",
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind address (use 0.0.0.0 to expose on LAN)")
    return p.parse_args()


def _load_model(checkpoint: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    logger.info("Loading Molmo2-4B from '%s'…", checkpoint)

    quant_cfg = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["vision_backbone", "lm_head"],
            llm_int8_enable_fp32_cpu_offload=True,
        )
        if torch.cuda.is_available()
        else None
    )

    load_kwargs: dict = dict(
        trust_remote_code=True,
        device_map="auto",
        max_memory={0: "4GiB", "cpu": "48GiB"},
    )
    if quant_cfg is not None:
        load_kwargs["quantization_config"] = quant_cfg
    else:
        load_kwargs["dtype"] = "auto"

    model = AutoModelForImageTextToText.from_pretrained(checkpoint, **load_kwargs)
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=True, use_fast=True
    )

    exec_device = next(
        (p.device for p in model.parameters() if p.device.type == "cuda"),
        torch.device("cpu"),
    )
    logger.info("Molmo2-4B ready on %s", exec_device)
    return model, processor, exec_device


def _run_inference(model, processor, exec_device, prompt: str, pil_image) -> str:
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": pil_image},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {
        k: (
            v.to(device=exec_device, dtype=torch.bfloat16)
            if v.is_floating_point()
            else v.to(exec_device)
        )
        for k, v in inputs.items()
    }

    torch.cuda.empty_cache()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=200)

    generated_tokens = generated_ids[0, inputs["input_ids"].size(1):]
    return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main() -> None:
    args = _parse_args()

    try:
        from flask import Flask, jsonify, request
    except ImportError:
        logger.error("flask is required: pip install flask")
        sys.exit(1)

    model, processor, exec_device = _load_model(args.checkpoint)

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/point", methods=["POST"])
    def point():
        data = request.get_json(force=True)
        prompt = data.get("prompt", "")
        image_b64 = data.get("image_b64", "")
        if not prompt or not image_b64:
            return jsonify({"error": "prompt and image_b64 are required"}), 400

        try:
            from PIL import Image
            img_bytes = base64.b64decode(image_b64)
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = pil_image.size
        except Exception as exc:
            return jsonify({"error": f"image decode failed: {exc}"}), 400

        try:
            text = _run_inference(model, processor, exec_device, prompt, pil_image)
        except Exception as exc:
            logger.exception("Inference failed")
            return jsonify({"error": f"inference failed: {exc}"}), 500

        logger.info("point | prompt=%r | output=%r", prompt[:80], text[:120])
        return jsonify({"text": text, "image_w": w, "image_h": h})

    logger.info("Starting Molmo server on %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
