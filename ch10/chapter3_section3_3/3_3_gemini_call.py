from __future__ import annotations

import os
from typing import Optional


def call_gemini(prompt: str, model: str = "gemini-1.5-flash", api_key: Optional[str] = None) -> str:
    """Call a Gemini model for optional agent experimentation."""

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError("Install google-generativeai to use call_gemini().") from exc

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY before calling this function.")

    genai.configure(api_key=key)
    model_obj = genai.GenerativeModel(model)
    response = model_obj.generate_content(prompt)
    return getattr(response, "text", "") or ""


if __name__ == "__main__":
    print("This is an optional API wrapper. Set GEMINI_API_KEY to run it.")
