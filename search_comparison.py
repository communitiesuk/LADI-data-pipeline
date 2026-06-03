"""
Compare 4 search methods against embedded documents in Azure Postgres.

Methods:
  1. Embedding   — pgvector cosine similarity (embedding <=> query_vec)
  2. Full text   — ILIKE keyword match on document text column
  3. Summary     — ILIKE keyword match on summary column
  4. Title       — ILIKE keyword match on title column

Usage:
    python search_comparison.py "infrastructure funding statement"
    python search_comparison.py "housing strategy" --k 10

Env vars required:
    LADI_DB_PASSWORD          — Postgres password
    LADI_DB_HOST              — defaults to Azure sandbox host
    LADI_DB_USER              — defaults to Tobi
    LADI_DB_NAME              — defaults to postgres
    LADI_APIM_EMBEDDING_URL   — for embedding search only
    LADI_APIM_SUBSCRIPTION_KEY
"""
import argparse
import asyncio
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.environ.get(
    "LADI_DB_HOST",
    "psql-sbx-uks-aaai-server-001.postgres.database.azure.com",
)
DB_PORT = int(os.environ.get("LADI_DB_PORT", "5432"))
DB_USER = os.environ.get("LADI_DB_USER", "Tobi")
DB_NAME = os.environ.get("LADI_DB_NAME", "postgres")
DB_TABLE = os.environ.get("LADI_DB_TABLE", "embedding_sample")


def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=os.environ["LADI_DB_PASSWORD"],
        dbname=DB_NAME,
        sslmode="require",
    )


def embed_query(text: str) -> list[float]:
    """Embed a query string via Azure APIM."""
    from ladi.apim import build_embedding_client

    client = build_embedding_client()

    async def _embed():
        resp = await client.embeddings.create(
            model="text-embedding-3-large",
            input=[text],
        )
        return resp.data[0].embedding

    return asyncio.run(_embed())


def search_embedding(cur, query: str, k: int) -> list[dict]:
    """Embed the query then rank by pgvector cosine distance."""
    vec = embed_query(query)
    vec_str = "[" + ",".join(str(v) for v in vec) + "]"
    cur.execute(
        f"""
        SELECT title, authority, url,
               1 - (embedding <=> %s::vector) AS similarity
        FROM {DB_TABLE}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (vec_str, vec_str, k),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _keyword_score_expr(column: str, terms: list[str], query: str) -> tuple[str, list]:
    """
    Build a SQL scoring expression for keyword matching.

    Score = (fraction of terms matched) + 0.5 if exact phrase present.
    Returns (sql_fragment, params_list).
    """
    cases = " + ".join(
        f"CASE WHEN {column} ILIKE %s THEN 1.0/{len(terms)} ELSE 0 END"
        for _ in terms
    )
    params = [f"%{t}%" for t in terms]

    score_sql = f"(({cases}) + CASE WHEN {column} ILIKE %s THEN 0.5 ELSE 0 END)"
    params.append(f"%{query.lower()}%")
    return score_sql, params


def search_keyword(cur, query: str, column: str, k: int) -> list[dict]:
    """Rank docs by how many query terms appear in `column` via ILIKE."""
    terms = [t.lower() for t in query.split() if len(t) > 2]
    if not terms:
        return []

    score_sql, score_params = _keyword_score_expr(column, terms, query)
    where_clause = " OR ".join(f"COALESCE({column}, '') ILIKE %s" for _ in terms)
    where_params = [f"%{t}%" for t in terms]

    cur.execute(
        f"""
        SELECT title, authority, url, {score_sql} AS score
        FROM {DB_TABLE}
        WHERE {where_clause}
        ORDER BY score DESC
        LIMIT %s
        """,
        score_params + where_params + [k],
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def print_results(method_name: str, results: list[dict], score_key: str = "score"):
    print(f"\n  {'─'*64}")
    print(f"  {method_name}")
    print(f"  {'─'*64}")
    if not results:
        print("  (no matches)")
        return
    for rank, r in enumerate(results, 1):
        score = float(r.get(score_key) or 0)
        title = (r.get("title") or "N/A")[:55]
        authority = (r.get("authority") or "N/A")[:30]
        print(f"  #{rank}  score={score:.4f}  {title}")
        print(f"       authority={authority}")


def main():
    p = argparse.ArgumentParser(description="Compare 4 search methods via Postgres+pgvector")
    p.add_argument("query", help="Search query text")
    p.add_argument("--k", type=int, default=5, help="Number of results (default: 5)")
    p.add_argument("--no-embed", action="store_true", help="Skip embedding search (no APIM call)")
    args = p.parse_args()

    conn = get_connection()
    cur = conn.cursor()

    print(f"{'='*70}")
    print(f"QUERY: \"{args.query}\"")
    print(f"{'='*70}")

    results_emb = [] if args.no_embed else search_embedding(cur, args.query, args.k)
    results_text = search_keyword(cur, args.query, "document_text", args.k)
    results_summary = search_keyword(cur, args.query, "summary", args.k)
    results_title = search_keyword(cur, args.query, "title", args.k)

    if not args.no_embed:
        print_results("1. EMBEDDING  (pgvector <=> cosine similarity)", results_emb, "similarity")
    print_results("2. FULL TEXT  (ILIKE keyword match)", results_text)
    print_results("3. SUMMARY    (ILIKE keyword match)", results_summary)
    print_results("4. TITLE      (ILIKE keyword match)", results_title)

    print(f"\n{'='*70}")
    print("OVERLAP ANALYSIS")
    print(f"{'='*70}")
    emb_urls = {r["url"] for r in results_emb}
    text_urls = {r["url"] for r in results_text}
    sum_urls = {r["url"] for r in results_summary}
    title_urls = {r["url"] for r in results_title}

    if not args.no_embed:
        print(f"  Embedding ∩ Full text:  {len(emb_urls & text_urls)}/{args.k}")
        print(f"  Embedding ∩ Summary:    {len(emb_urls & sum_urls)}/{args.k}")
        print(f"  Embedding ∩ Title:      {len(emb_urls & title_urls)}/{args.k}")
    print(f"  Summary   ∩ Title:      {len(sum_urls & title_urls)}/{args.k}")
    if not args.no_embed:
        print(f"  All 4 agree:            {len(emb_urls & text_urls & sum_urls & title_urls)}/{args.k}")
    print()

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
