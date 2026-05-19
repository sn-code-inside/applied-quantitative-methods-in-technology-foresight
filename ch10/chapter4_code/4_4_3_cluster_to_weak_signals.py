from __future__ import annotations

from typing import Dict, List


CLUSTER_PROMPT = """Interpret the following cluster as a potential weak signal.
Return: (1) possible opportunity, (2) possible risk, (3) governance implication,
and (4) uncertainty note. Treat the output as a hypothesis for human validation.
"""


def cluster_to_weak_signal(cluster: Dict) -> Dict[str, str]:
    """Convert a cluster of snippets into a structured weak-signal hypothesis.

    This is a deterministic educational substitute for LLM summarization. The
    goal is to make the logic inspectable and reproducible.
    """

    snippets: List[str] = cluster.get("snippets", [])
    combined = " ".join(snippets)
    return {
        "cluster_id": str(cluster.get("cluster_id", "unknown")),
        "weak_signal_hypothesis": combined[:240] + ("..." if len(combined) > 240 else ""),
        "opportunity": "Shared interpretation may help SMEs anticipate changes earlier.",
        "risk": "The signal may be over-interpreted without expert validation.",
        "governance_implication": "Human-in-the-loop review is required before strategic action.",
        "uncertainty_note": "Educational output; replace with evidence-grounded LLM or expert synthesis in applied work.",
    }


if __name__ == "__main__":
    demo_cluster = {"cluster_id": 0, "snippets": ["Traceability costs are rising for SMEs."]}
    print(cluster_to_weak_signal(demo_cluster))
