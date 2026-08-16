from __future__ import annotations

import re

from ..config import load_models

MODEL_CONFIG = load_models()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = MODEL_CONFIG.get("openrouter", {}).get("default_generation_model", "qwen/qwen3.5-flash-02-23")

_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def generate(system: str, user: str, model: str | None = None, api_key: str | None = None) -> str:
    """Call the configured OpenRouter model with the given prompts; returns assistant text."""
    import os

    from openai import OpenAI

    key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is unset. Set it in .env (or pass --offline to use mock candidates)."
        )
    client = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)
    resp = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def extract_code(text: str) -> str:
    """Extract the first ```python ... ``` fenced block wherever it appears in the
    response (the model may wrap it in prose); fall back to the raw text otherwise."""
    stripped = text.strip()
    m = _FENCE.search(stripped)
    if m:
        return m.group(1).strip()
    return stripped
