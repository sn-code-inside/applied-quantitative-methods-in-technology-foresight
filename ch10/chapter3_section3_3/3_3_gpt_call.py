from __future__ import annotations

import os
from typing import Optional


def call_gpt(prompt: str, model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> str:
    """Call an OpenAI-compatible chat model.

    This is optional. The main repository pipeline runs offline. Use this only
    if you want to reproduce API-based experimentation and have a valid key.
    """

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the openai package to use call_gpt().") from exc

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY before calling this function.")

    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a foresight analyst supporting anticipatory governance."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    print("This is an optional API wrapper. Set OPENAI_API_KEY to run it.")
