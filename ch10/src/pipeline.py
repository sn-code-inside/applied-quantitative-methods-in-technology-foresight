from __future__ import annotations

import argparse
import json
from pathlib import Path

from .text_io import load_documents
from .retrieval import build_simple_index, top_k
from .towers import make_tower1, make_tower2, make_tower3


def main() -> None:
    """Run the simplified three-tower weak-signal pipeline.

    Example
    -------
    python -m src.pipeline --input data/sample_docs --out out
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder with .txt documents")
    parser.add_argument("--out", default="out", help="Output folder")
    parser.add_argument("--k", type=int, default=6, help="Top-K chunks to retrieve")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents(input_dir)
    index = build_simple_index(docs)

    # Educational query derived from the chapter topic. In classroom use,
    # students can replace it with their own domain-specific weak-signal focus.
    query = "weak signals for SMEs governance foresight networks traceability"
    retrieved = top_k(index, query=query, k=args.k)

    tower1 = make_tower1(retrieved)
    tower2 = make_tower2(retrieved)
    tower3 = make_tower3(retrieved)

    (output_dir / "tower1.md").write_text(tower1, encoding="utf-8")
    (output_dir / "tower2.md").write_text(tower2, encoding="utf-8")
    (output_dir / "tower3.json").write_text(
        json.dumps(tower3, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
