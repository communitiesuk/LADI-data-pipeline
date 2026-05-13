"""
Evaluate embedding quality on the summaries_sample dataset.

Metrics:
  1. Sanity — dimensions, norms, zero/NaN vectors
  2. Theme coherence — avg cosine sim within-theme vs between-theme
  3. k-NN theme precision — % of top-k neighbors sharing ≥1 theme
  4. Authority coherence — avg cosine sim within-authority vs between-authority

Usage:
    python eval_embedding.py --input data/embeddings_sample.jsonl
"""
import argparse
import json
import ast
from collections import defaultdict

import numpy as np


def load_embeddings(path: str) -> list[dict]:
    """Load records with embeddings from JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "embedding" not in obj:
                continue
            records.append(obj)
    return records


def parse_themes(themes_field) -> list[str]:
    """Parse themes from various formats (list, string repr of list)."""
    if isinstance(themes_field, list):
        return themes_field
    if isinstance(themes_field, str):
        try:
            parsed = ast.literal_eval(themes_field)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        return [themes_field]
    return []


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)  # avoid division by zero
    normed = embeddings / norms
    return normed @ normed.T


def sanity_checks(embeddings: np.ndarray) -> dict:
    """Basic sanity checks on embedding vectors."""
    n, dims = embeddings.shape
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        "n_docs": n,
        "dimensions": dims,
        "zero_vectors": int(np.sum(norms < 1e-8)),
        "nan_vectors": int(np.sum(np.isnan(norms))),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
    }


def theme_coherence(sim_matrix: np.ndarray, themes_per_doc: list[list[str]]) -> dict:
    """Compute within-theme vs between-theme avg cosine similarity."""
    n = len(themes_per_doc)

    # Build theme → doc indices mapping
    theme_docs = defaultdict(set)
    for i, themes in enumerate(themes_per_doc):
        for t in themes:
            theme_docs[t].add(i)

    within_sims = []
    between_sims = []

    for theme, indices in theme_docs.items():
        indices = sorted(indices)
        if len(indices) < 2:
            continue
        # Within-theme pairs
        for i_idx in range(len(indices)):
            for j_idx in range(i_idx + 1, len(indices)):
                within_sims.append(sim_matrix[indices[i_idx], indices[j_idx]])

    # Between-theme: sample random pairs not sharing any theme
    rng = np.random.default_rng(42)
    n_between_samples = min(len(within_sims) * 2, n * (n - 1) // 2)
    sampled = 0
    attempts = 0
    while sampled < n_between_samples and attempts < n_between_samples * 10:
        i, j = rng.integers(0, n, size=2)
        if i == j:
            attempts += 1
            continue
        shared = set(themes_per_doc[i]) & set(themes_per_doc[j])
        if not shared:
            between_sims.append(sim_matrix[i, j])
            sampled += 1
        attempts += 1

    within_mean = float(np.mean(within_sims)) if within_sims else 0.0
    between_mean = float(np.mean(between_sims)) if between_sims else 0.0

    # Per-theme breakdown
    per_theme = {}
    for theme, indices in sorted(theme_docs.items()):
        indices = sorted(indices)
        if len(indices) < 2:
            per_theme[theme] = {"n_docs": len(indices), "avg_sim": None}
            continue
        sims = []
        for i_idx in range(len(indices)):
            for j_idx in range(i_idx + 1, len(indices)):
                sims.append(sim_matrix[indices[i_idx], indices[j_idx]])
        per_theme[theme] = {"n_docs": len(indices), "avg_sim": float(np.mean(sims))}

    return {
        "within_theme_sim": within_mean,
        "between_theme_sim": between_mean,
        "coherence_ratio": within_mean / between_mean if between_mean > 0 else float("inf"),
        "n_within_pairs": len(within_sims),
        "n_between_pairs": len(between_sims),
        "per_theme": per_theme,
    }


def knn_theme_precision(sim_matrix: np.ndarray, themes_per_doc: list[list[str]], k: int = 5) -> dict:
    """For each doc, what fraction of its k nearest neighbors share ≥1 theme?"""
    n = sim_matrix.shape[0]
    precisions = []

    for i in range(n):
        # Get top-k neighbors (excluding self)
        sims = sim_matrix[i].copy()
        sims[i] = -1  # exclude self
        top_k_indices = np.argsort(sims)[-k:]

        # Check theme overlap
        my_themes = set(themes_per_doc[i])
        if not my_themes:
            continue
        hits = sum(1 for j in top_k_indices if set(themes_per_doc[j]) & my_themes)
        precisions.append(hits / k)

    return {
        "k": k,
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "median_precision": float(np.median(precisions)) if precisions else 0.0,
        "min_precision": float(np.min(precisions)) if precisions else 0.0,
        "n_evaluated": len(precisions),
    }


def authority_coherence(sim_matrix: np.ndarray, authorities: list[str]) -> dict:
    """Avg cosine sim within-authority vs between-authority."""
    auth_docs = defaultdict(list)
    for i, auth in enumerate(authorities):
        auth_docs[auth].append(i)

    within_sims = []
    for auth, indices in auth_docs.items():
        if len(indices) < 2:
            continue
        for i_idx in range(len(indices)):
            for j_idx in range(i_idx + 1, len(indices)):
                within_sims.append(sim_matrix[indices[i_idx], indices[j_idx]])

    # Sample between-authority pairs
    n = sim_matrix.shape[0]
    rng = np.random.default_rng(42)
    between_sims = []
    n_samples = min(len(within_sims) * 2, 10000)
    sampled = 0
    attempts = 0
    while sampled < n_samples and attempts < n_samples * 10:
        i, j = rng.integers(0, n, size=2)
        if i == j or authorities[i] == authorities[j]:
            attempts += 1
            continue
        between_sims.append(sim_matrix[i, j])
        sampled += 1
        attempts += 1

    within_mean = float(np.mean(within_sims)) if within_sims else 0.0
    between_mean = float(np.mean(between_sims)) if between_sims else 0.0

    return {
        "within_authority_sim": within_mean,
        "between_authority_sim": between_mean,
        "coherence_ratio": within_mean / between_mean if between_mean > 0 else float("inf"),
        "n_authorities": len(auth_docs),
        "n_within_pairs": len(within_sims),
    }


def print_report(sanity: dict, theme_coh: dict, knn: dict, auth_coh: dict) -> None:
    """Print formatted evaluation report."""
    print("=" * 70)
    print("EMBEDDING EVALUATION REPORT")
    print("=" * 70)

    # Sanity
    print("\n1. SANITY CHECKS")
    print("-" * 40)
    print(f"   Documents:       {sanity['n_docs']}")
    print(f"   Dimensions:      {sanity['dimensions']}")
    print(f"   Zero vectors:    {sanity['zero_vectors']}")
    print(f"   NaN vectors:     {sanity['nan_vectors']}")
    print(f"   Norm (mean/std): {sanity['norm_mean']:.4f} / {sanity['norm_std']:.4f}")
    print(f"   Norm (min/max):  {sanity['norm_min']:.4f} / {sanity['norm_max']:.4f}")
    ok = sanity["zero_vectors"] == 0 and sanity["nan_vectors"] == 0 and sanity["dimensions"] == 3072
    print(f"   Status:          {'PASS' if ok else 'FAIL'}")

    # Theme coherence
    print("\n2. THEME COHERENCE")
    print("-" * 40)
    print(f"   Within-theme avg sim:  {theme_coh['within_theme_sim']:.4f}")
    print(f"   Between-theme avg sim: {theme_coh['between_theme_sim']:.4f}")
    print(f"   Coherence ratio:       {theme_coh['coherence_ratio']:.3f}x")
    print(f"   Pairs (within/between): {theme_coh['n_within_pairs']:,} / {theme_coh['n_between_pairs']:,}")
    ok = theme_coh["coherence_ratio"] > 1.0
    print(f"   Status:          {'PASS' if ok else 'FAIL'} (ratio > 1.0 required)")

    print("\n   Per-theme breakdown:")
    for theme, data in sorted(theme_coh["per_theme"].items(), key=lambda x: -(x[1]["avg_sim"] or 0)):
        sim_str = f"{data['avg_sim']:.4f}" if data["avg_sim"] is not None else "N/A (1 doc)"
        print(f"     {theme:<35} n={data['n_docs']:<4} avg_sim={sim_str}")

    # k-NN
    print(f"\n3. k-NN THEME PRECISION (k={knn['k']})")
    print("-" * 40)
    print(f"   Mean precision:   {knn['mean_precision']:.4f}")
    print(f"   Median precision: {knn['median_precision']:.4f}")
    print(f"   Min precision:    {knn['min_precision']:.4f}")
    print(f"   Docs evaluated:   {knn['n_evaluated']}")
    # Random baseline depends on theme distribution, but typically ~0.3-0.4
    ok = knn["mean_precision"] > 0.5
    print(f"   Status:          {'PASS' if ok else 'FAIL'} (mean > 0.50 required)")

    # Authority coherence
    print("\n4. AUTHORITY COHERENCE")
    print("-" * 40)
    print(f"   Within-authority avg sim:  {auth_coh['within_authority_sim']:.4f}")
    print(f"   Between-authority avg sim: {auth_coh['between_authority_sim']:.4f}")
    print(f"   Coherence ratio:           {auth_coh['coherence_ratio']:.3f}x")
    print(f"   Authorities:               {auth_coh['n_authorities']}")
    ok = auth_coh["coherence_ratio"] > 1.0
    print(f"   Status:          {'PASS' if ok else 'FAIL'} (ratio > 1.0 required)")

    print("\n" + "=" * 70)
    all_pass = (
        sanity["zero_vectors"] == 0
        and sanity["nan_vectors"] == 0
        and theme_coh["coherence_ratio"] > 1.0
        and knn["mean_precision"] > 0.5
        and auth_coh["coherence_ratio"] > 1.0
    )
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 70)


def main():
    p = argparse.ArgumentParser(description="Evaluate embedding quality")
    p.add_argument("--input", default="data/embeddings_sample.jsonl",
                   help="Embeddings JSONL file to evaluate")
    p.add_argument("--k", type=int, default=5, help="k for k-NN precision (default: 5)")
    args = p.parse_args()

    print(f"Loading {args.input} ...")
    records = load_embeddings(args.input)
    print(f"Loaded {len(records)} embedded documents")

    embeddings = np.array([r["embedding"] for r in records])
    themes_per_doc = [parse_themes(r.get("themes", [])) for r in records]
    authorities = [r.get("authority", "") for r in records]

    print("Computing cosine similarity matrix ...")
    sim_matrix = cosine_similarity_matrix(embeddings)

    sanity = sanity_checks(embeddings)
    theme_coh = theme_coherence(sim_matrix, themes_per_doc)
    knn = knn_theme_precision(sim_matrix, themes_per_doc, k=args.k)
    auth_coh = authority_coherence(sim_matrix, authorities)

    print_report(sanity, theme_coh, knn, auth_coh)


if __name__ == "__main__":
    main()
