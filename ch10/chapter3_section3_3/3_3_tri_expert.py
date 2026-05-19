from __future__ import annotations

from typing import Dict
import json


SECURITY_PROMPT = """You are an agrifood security expert with a conservative innovation perspective.
For the weak signal provided, identify one opportunity, one risk, and one governance implication.
"""

MARKET_PROMPT = """You are an agrifood market expert with a disruptive innovation perspective.
For the weak signal provided, identify one opportunity, one risk, and one governance implication.
"""

REGULATORY_PROMPT = """You are an agrifood law and regulation expert with a moderate innovation perspective.
For the weak signal provided, identify one opportunity, one risk, and one governance implication.
"""


def interpret_offline(signal: str, role: str) -> Dict[str, str]:
    """Offline deterministic substitute for an LLM role interpretation.

    The function is designed for reproducibility in the Springer repository.
    In classroom use, instructors can replace this with `call_gpt` or
    `call_gemini` from the companion scripts.
    """

    base = {
        "signal": signal,
        "role": role,
        "opportunity": f"{role} detects a possible strategic opportunity associated with {signal}.",
        "risk_or_warning": f"{role} notes uncertainty and possible unintended consequences.",
        "governance_implication": "Human experts should compare perspectives before policy or strategy decisions.",
    }
    return base


def run_tri_expert(signal: str) -> Dict[str, Dict[str, str]]:
    """Run a three-perspective interpretation of a weak signal."""

    return {
        "security": interpret_offline(signal, "Agrifood security expert"),
        "market": interpret_offline(signal, "Agrifood market expert"),
        "regulatory": interpret_offline(signal, "Agrifood regulation expert"),
    }


if __name__ == "__main__":
    result = run_tri_expert("Cultivated meat")
    print(json.dumps(result, indent=2, ensure_ascii=False))
