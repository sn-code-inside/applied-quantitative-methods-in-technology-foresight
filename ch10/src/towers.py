from __future__ import annotations

from typing import Any, Dict, List

from .retrieval import Chunk


def make_tower1(chunks: List[Chunk]) -> str:
    """Create Tower 1: a short strategic insight.

    In a production system, this layer may use LLM-supported synthesis. Here we
    use a deterministic educational template so readers can run the example
    offline and inspect the logic.
    """

    evidence = "\n".join(
        f"- {chunk.title} ({chunk.date}): {chunk.text.strip()[:150]}..."
        for chunk in chunks[:3]
    )

    return (
        "# Tower 1 — Strategic Insight\n\n"
        "**Insight (educational demo):** SME networks can democratize foresight "
        "by combining AI-assisted pre-processing with human collective "
        "sensemaking. The three-tower structure enables fast scanning (Tower 1), "
        "explanatory narratives (Tower 2), and evidence verification (Tower 3), "
        "reducing attention scarcity while preserving interpretive responsibility.\n\n"
        "## Evidence highlights\n"
        f"{evidence}\n"
    )


def make_tower2(chunks: List[Chunk]) -> str:
    """Create Tower 2: thematic contexts and intermediate interpretation."""

    return """# Tower 2 — Thematic Contexts

## Context A — Resource scarcity and attention scarcity
SMEs face constraints that limit dedicated scanning capacity. A SCALE-inspired pipeline reduces manual effort by structuring information hierarchically while keeping humans responsible for interpretation.

## Context B — Network collaboration and shared intelligence
Inter-organizational collaboration distributes scanning burdens and diversifies perspectives, strengthening collective sensemaking and reducing over-reliance on any single expert.

## Context C — Traceability and validation
Each strategic claim must remain linked to traceable fragments (Tower 3) to enable verification, debate, and responsible governance.
"""


def make_tower3(chunks: List[Chunk]) -> Dict[str, Any]:
    """Create Tower 3: traceable documentary fragments."""

    fragments = []
    for chunk in chunks:
        fragments.append(
            {
                "fragment_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "date": chunk.date,
                "source": chunk.source,
                "license": chunk.license,
                "excerpt": chunk.text.strip()[:520],
            }
        )
    return {
        "fragments": fragments,
        "note": "Educational sample: students should replace synthetic samples with properly licensed sources and add URLs/DOIs when available.",
    }
