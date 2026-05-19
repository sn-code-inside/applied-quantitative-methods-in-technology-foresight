from __future__ import annotations

from typing import List, Dict, Any


def cluster_snippets(snippets: List[str], n_clusters: int = 3) -> List[Dict[str, Any]]:
    """Cluster snippets using sentence embeddings and k-means.

    This script requires `sentence-transformers` and `scikit-learn`. It is an
    educational demonstration of how weak-signal fragments can be grouped before
    human interpretation.
    """

    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    if not snippets:
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(snippets)
    n_clusters = min(n_clusters, len(snippets))
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(embeddings)

    clusters: List[Dict[str, Any]] = []
    for label in sorted(set(labels)):
        clusters.append(
            {
                "cluster_id": int(label),
                "snippets": [snippet for snippet, assigned in zip(snippets, labels) if assigned == label],
            }
        )
    return clusters


if __name__ == "__main__":
    snippets = [
        "SMEs share compliance intelligence through regional networks.",
        "New traceability rules increase documentation costs.",
        "Collective foresight sessions help firms interpret weak signals.",
    ]
    print(cluster_snippets(snippets, n_clusters=2))
