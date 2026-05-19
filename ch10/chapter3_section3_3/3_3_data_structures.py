from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List
import json


@dataclass
class WeakSignal:
    """Minimal representation of a weak signal for educational examples."""

    name: str
    short_description: str
    source: str
    uncertainty_note: str


@dataclass
class AgentOutput:
    """Structured output expected from a specialized foresight agent."""

    agent_role: str
    opportunity: str
    risk_or_warning: str
    governance_implication: str
    confidence: str


def to_jsonable(obj: Any) -> Dict[str, Any]:
    """Convert dataclass objects to dictionaries for JSON export."""
    return asdict(obj)


if __name__ == "__main__":
    signal = WeakSignal(
        name="Cultivated meat",
        short_description="Alternative protein technology based on cell culture.",
        source="JRC weak-signal report / educational example",
        uncertainty_note="Market adoption and regulatory acceptance remain uncertain.",
    )
    output = AgentOutput(
        agent_role="Agrifood regulation expert",
        opportunity="Definition of new standards for safety, labeling, and trade.",
        risk_or_warning="Fragmented regulation may delay diffusion.",
        governance_implication="Early policy coordination can reduce uncertainty.",
        confidence="medium",
    )
    print(json.dumps({"signal": to_jsonable(signal), "output": to_jsonable(output)}, indent=2))
