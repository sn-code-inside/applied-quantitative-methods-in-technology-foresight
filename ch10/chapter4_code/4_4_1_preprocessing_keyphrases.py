from __future__ import annotations

from collections import Counter
import re
from typing import List, Tuple


def normalize(text: str) -> str:
    """Normalize whitespace and lowercase text for simple analysis."""
    text = re.sub(r"\s+", " ", text.strip())
    return text.lower()


def simple_keyphrases(text: str, top_n: int = 10) -> List[Tuple[str, int]]:
    """Extract simple frequency-based keyphrases.

    This is intentionally simple for teaching. In a richer implementation, this
    step could be replaced by RAKE, KeyBERT, BERTopic, or LLM-assisted tagging.
    """

    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "are", "into",
        "can", "may", "not", "but", "their", "about", "using", "used",
    }
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", normalize(text))
    filtered = [word for word in words if word not in stopwords]
    return Counter(filtered).most_common(top_n)


if __name__ == "__main__":
    sample = "SME networks can share foresight resources and detect weak signals earlier."
    print(simple_keyphrases(sample))
