"""
Quick search demo against embedded documents.

Embeds a query string, finds top-k most similar documents, and prints results.

Usage:
    python search_demo.py "housing strategy affordable homes"
    python search_demo.py "council budget cuts" --k 10
    python search_demo.py --interactive
"""
import argparse
import json

import numpy as np
from dotenv import load_dotenv

load_dotenv()


def load_embeddings(path: str) -> tuple[list[dict], np.ndarray]:
    """Load records and embedding matrix."""
    records = []
    vecs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "embedding" not in obj:
                continue
            vecs.append(obj["embedding"])
            records.append(obj)
    return records, np.array(vecs)


def embed_query(text: str) -> np.ndarray:
    """Embed a query string using the APIM embedding client."""
    from ladi.apim import build_embedding_client
    import asyncio

    client = build_embedding_client()

    async def _embed():
        resp = await client.embeddings.create(
            model="text-embedding-3-large",
            input=[text],
        )
        return resp.data[0].embedding

    return np.array(asyncio.run(_embed()))


def search(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int) -> list[int]:
    """Return indices of top-k most similar documents."""
    # Cosine similarity (vectors are already unit-normed)
    query_norm = query_vec / np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    doc_normed = doc_vecs / np.clip(doc_norms, 1e-10, None)
    sims = doc_normed @ query_norm
    return np.argsort(sims)[-k:][::-1].tolist(), sims


def print_results(records: list[dict], indices: list[int], sims: np.ndarray, query: str):
    """Print search results."""
    print(f"\n{'='*70}")
    print(f"QUERY: \"{query}\"")
    print(f"{'='*70}")
    for rank, idx in enumerate(indices, 1):
        r = records[idx]
        print(f"\n  #{rank} (sim={sims[idx]:.4f})")
        print(f"  Title:     {r.get('title', 'N/A')}")
        print(f"  Authority: {r.get('authority', 'N/A')}")
        print(f"  Themes:    {r.get('themes', 'N/A')}")
        print(f"  Summary:   {r.get('summary', 'N/A')[:150]}...")
    print()


def main():
    p = argparse.ArgumentParser(description="Search embedded documents")
    p.add_argument("query", nargs="?", help="Search query text")
    p.add_argument("--input", default="data/embeddings_sample.jsonl")
    p.add_argument("--k", type=int, default=5, help="Number of results (default: 5)")
    p.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = p.parse_args()

    print(f"Loading embeddings from {args.input} ...")
    records, doc_vecs = load_embeddings(args.input)
    print(f"Loaded {len(records)} documents")

    if args.interactive:
        print("\nEnter queries (Ctrl+C to quit):\n")
        while True:
            try:
                query = input("Search> ").strip()
                if not query:
                    continue
                query_vec = embed_query(query)
                indices, sims = search(query_vec, doc_vecs, args.k)
                print_results(records, indices, sims, query)
            except (KeyboardInterrupt, EOFError):
                print("\nDone.")
                break
    else:
        if not args.query:
            p.error("Provide a query or use --interactive")
        query_vec = embed_query(args.query)
        indices, sims = search(query_vec, doc_vecs, args.k)
        print_results(records, indices, sims, args.query)


if __name__ == "__main__":
    main()
