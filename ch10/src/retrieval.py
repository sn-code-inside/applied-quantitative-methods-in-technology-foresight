from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import math
import re

from .text_io import Document


def _tokenize(text: str) -> List[str]:
    """Very small tokenizer used only for the educational demo."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


@dataclass
class Chunk:
    """A traceable fragment from an input document."""

    doc_id: str
    title: str
    date: str
    source: str
    license: str
    chunk_id: str
    text: str
    tf: Dict[str, int]


def build_simple_index(docs: List[Document], chunk_size: int = 450) -> List[Chunk]:
    """Split documents into chunks and build a simple term-frequency index.

    This deliberately avoids restricted APIs or black-box infrastructure. It is
    not meant to replace semantic search in production; it is a transparent
    teaching implementation.
    """

    chunks: List[Chunk] = []
    for doc in docs:
        text = doc.text.strip()
        parts = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        for i, part in enumerate(parts):
            tf: Dict[str, int] = {}
            for token in _tokenize(part):
                tf[token] = tf.get(token, 0) + 1
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    date=doc.date,
                    source=doc.source,
                    license=doc.license,
                    chunk_id=f"{doc.doc_id}:{i}",
                    text=part,
                    tf=tf,
                )
            )
    return chunks


def _cosine(a: Dict[str, int], b: Dict[str, int]) -> float:
    """Sparse cosine similarity over term-frequency dictionaries."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for key, value in a.items():
        norm_a += value * value
        if key in b:
            dot += value * b[key]
    for value in b.values():
        norm_b += value * value
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def top_k(index: List[Chunk], query: str, k: int = 6) -> List[Chunk]:
    """Return the top-k chunks most similar to the query."""
    qtf: Dict[str, int] = {}
    for token in _tokenize(query):
        qtf[token] = qtf.get(token, 0) + 1

    scored = [(_cosine(chunk.tf, qtf), chunk) for chunk in index]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:k]]
