"""
Compare 4 search methods against the embedded document set.

Methods:
  1. Embedding (cosine similarity on summary embedding)
  2. Full text (keyword match on document text)
  3. Summary text (keyword match on summary)
  4. Title (keyword match on title)

Usage:
    python search_comparison.py "infrastructure funding statement"
    python search_comparison.py "housing strategy" --k 10
"""
import argparse
import asyncio
import json
import re
from collections import Counter

import numpy as np
from dotenv import load_dotenv

load_dotenv()


def load_records(path: str) -> list[dict]:
    """Load all records from embeddings JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "embedding" in obj:
                records.append(obj)
    return records


def embed_query(text: str) -> np.ndarray:
    """Embed a query string."""
    from ladi.apim import build_embedding_client

    client = build_embedding_client()

    async def _embed():
        resp = await client.embeddings.create(
            model="text-embedding-3-large",
            input=[text],
        )
        return resp.data[0].embedding

    return np.array(asyncio.run(_embed()))


def search_embedding(query: str, records: list[dict], doc_vecs: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Method 1: Cosine similarity on embeddings."""
    query_vec = embed_query(query)
    query_norm = query_vec / np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    doc_normed = doc_vecs / np.clip(doc_norms, 1e-10, None)
    sims = doc_normed @ query_norm
    top_k = np.argsort(sims)[-k:][::-1]
    return [(idx, float(sims[idx])) for idx in top_k]


def keyword_score(query: str, text: str) -> float:
    """Simple keyword matching score: fraction of query terms found in text."""
    if not text:
        return 0.0
    query_terms = [t.lower() for t in query.split() if len(t) > 2]
    text_lower = text.lower()
    hits = sum(1 for term in query_terms if term in text_lower)
    base_score = hits / len(query_terms) if query_terms else 0.0

    # Boost for exact phrase match
    if query.lower() in text_lower:
        base_score = min(base_score + 0.5, 1.0)

    # Boost by frequency of matches (normalized)
    total_matches = sum(text_lower.count(term) for term in query_terms)
    freq_boost = min(total_matches / 20.0, 0.3)

    return base_score + freq_boost


def search_text(query: str, records: list[dict], k: int) -> list[tuple[int, float]]:
    """Method 2: Keyword match on full document text."""
    scores = [(i, keyword_score(query, r.get("text", ""))) for i, r in enumerate(records)]
    scores.sort(key=lambda x: -x[1])
    return scores[:k]


def search_summary(query: str, records: list[dict], k: int) -> list[tuple[int, float]]:
    """Method 3: Keyword match on summary."""
    scores = [(i, keyword_score(query, r.get("summary", ""))) for i, r in enumerate(records)]
    scores.sort(key=lambda x: -x[1])
    return scores[:k]


def search_title(query: str, records: list[dict], k: int) -> list[tuple[int, float]]:
    """Method 4: Keyword match on title."""
    scores = [(i, keyword_score(query, r.get("title", ""))) for i, r in enumerate(records)]
    scores.sort(key=lambda x: -x[1])
    return scores[:k]


def print_results(method_name: str, results: list[tuple[int, float]], records: list[dict]):
    """Print search results for one method."""
    print(f"\n  {'─'*64}")
    print(f"  {method_name}")
    print(f"  {'─'*64}")
    for rank, (idx, score) in enumerate(results, 1):
        r = records[idx]
        title = r.get("title", "N/A")[:55]
        authority = r.get("authority", "N/A")[:25]
        print(f"  #{rank}  score={score:.4f}  {title}")
        print(f"      authority={authority}")
    if not results or results[0][1] == 0:
        print(f"  (no matches)")


def main():
    p = argparse.ArgumentParser(description="Compare 4 search methods")
    p.add_argument("query", help="Search query text")
    p.add_argument("--input", default="data/embeddings_combined.jsonl")
    p.add_argument("--k", type=int, default=5, help="Number of results (default: 5)")
    args = p.parse_args()

    print(f"Loading {args.input} ...")
    records = load_records(args.input)
    doc_vecs = np.array([r["embedding"] for r in records])
    print(f"Loaded {len(records)} documents\n")

    print(f"{'='*70}")
    print(f"QUERY: \"{args.query}\"")
    print(f"{'='*70}")

    # Run all 4 methods
    results_emb = search_embedding(args.query, records, doc_vecs, args.k)
    results_text = search_text(args.query, records, args.k)
    results_summary = search_summary(args.query, records, args.k)
    results_title = search_title(args.query, records, args.k)

    print_results("1. EMBEDDING (cosine similarity)", results_emb, records)
    print_results("2. FULL TEXT (keyword match)", results_text, records)
    print_results("3. SUMMARY (keyword match)", results_summary, records)
    print_results("4. TITLE (keyword match)", results_title, records)

    # Summary comparison
    print(f"\n{'='*70}")
    print("OVERLAP ANALYSIS")
    print(f"{'='*70}")
    emb_urls = {records[i]["url"] for i, _ in results_emb}
    text_urls = {records[i]["url"] for i, _ in results_text}
    summary_urls = {records[i]["url"] for i, _ in results_summary}
    title_urls = {records[i]["url"] for i, _ in results_title}

    print(f"  Embedding ∩ Full text:  {len(emb_urls & text_urls)}/{args.k}")
    print(f"  Embedding ∩ Summary:    {len(emb_urls & summary_urls)}/{args.k}")
    print(f"  Embedding ∩ Title:      {len(emb_urls & title_urls)}/{args.k}")
    print(f"  Full text ∩ Summary:    {len(text_urls & summary_urls)}/{args.k}")
    print(f"  Full text ∩ Title:      {len(text_urls & title_urls)}/{args.k}")
    print(f"  Summary ∩ Title:        {len(summary_urls & title_urls)}/{args.k}")
    print(f"  All 4 agree:            {len(emb_urls & text_urls & summary_urls & title_urls)}/{args.k}")
    print()


if __name__ == "__main__":
    main()
