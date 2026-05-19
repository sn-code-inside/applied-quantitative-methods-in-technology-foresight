from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from ch10.src.text_io import load_documents
from ch10.src.retrieval import build_simple_index, top_k
from ch10.src.towers import make_tower1, make_tower2, make_tower3


def run_pipeline(input_folder: str = "data/sample_docs", output_folder: str = "out") -> Dict[str, str]:
    """Run a minimal end-to-end educational weak-signal pipeline."""

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    docs = load_documents(input_path)
    index = build_simple_index(docs)
    retrieved = top_k(index, query="SME foresight weak signals traceability", k=6)

    tower1 = make_tower1(retrieved)
    tower2 = make_tower2(retrieved)
    tower3 = make_tower3(retrieved)

    (output_path / "tower1.md").write_text(tower1, encoding="utf-8")
    (output_path / "tower2.md").write_text(tower2, encoding="utf-8")
    (output_path / "tower3.json").write_text(json.dumps(tower3, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "tower1": str(output_path / "tower1.md"),
        "tower2": str(output_path / "tower2.md"),
        "tower3": str(output_path / "tower3.json"),
    }


if __name__ == "__main__":
    print(run_pipeline())
