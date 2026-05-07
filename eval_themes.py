"""Evaluate theme classification against manually tagged docs."""
import asyncio
import json
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

THEMES = [
    "Advice and benefits",
    "Adult social care",
    "Business and employment",
    "Community safety",
    "Council and democracy",
    "Children and young families",
    "Education",
    "Environment and waste",
    "Housing",
    "Leisure and culture",
    "Local government finance",
    "People and communities",
    "Planning and development",
    "Public health",
    "Transport and highways",
    "Uncategorised",
]

SYSTEM_PROMPT = f"""You classify UK local authority documents into themes.

Given document text, return JSON with:
- themes: array of 1-2 themes from the list below

Only give 2 themes if the document genuinely fits both — not just because it mentions a related topic. For example, a referendum on a neighbourhood plan is both "Council and democracy" and "Planning and development" because it is genuinely a democratic process about a planning matter.

Themes:
{chr(10).join(f'- {t}' for t in THEMES)}

Respond with valid JSON only. No markdown fences."""


def build_client() -> AsyncOpenAI:
    from ladi.apim import _get_token
    return AsyncOpenAI(
        base_url=os.environ["LADI_APIM_BASE_URL"],
        api_key=_get_token(),
        default_headers={"Ocp-Apim-Subscription-Key": os.environ["LADI_APIM_SUBSCRIPTION_KEY"]},
        default_query={"api-version": os.environ["LADI_APIM_API_VERSION"]},
        max_retries=0,
        timeout=10.0,
    )


def clean_text(text: str) -> str:
    """Strip invisible control characters that cause APIM 403s."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


def truncate_words(text: str, max_words: int = 1000) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


async def classify(client: AsyncOpenAI, text: str) -> dict:
    resp = await client.chat.completions.create(
        model="gpt-4-1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": clean_text(truncate_words(text, 750))},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=100,
    )
    return json.loads(resp.choices[0].message.content)


async def main():
    run_start = time.time()

    df = pd.read_csv("manually_tagged_docs.csv")
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.split().str.len() >= 50]
    total = len(df)
    print(f"Evaluating {total} docs\n", flush=True)

    client = build_client()
    results = []

    for i, (_, row) in enumerate(df.iterrows()):
        start = time.time()
        try:
            pred = await classify(client, row["text"])
        except Exception as e:
            if "429" in str(e):
                print(f"    ⏳ [{i+1}/{total}] Rate limited — waiting 60s...", flush=True)
                await asyncio.sleep(60)
                try:
                    pred = await classify(client, row["text"])
                except Exception as e2:
                    print(f"  ERROR [{i+1}/{total}] retry failed: {str(e2)[:80]}", flush=True)
                    continue
            else:
                print(f"  ERROR [{i+1}/{total}] {type(e).__name__}: {str(e)[:80]}", flush=True)
                continue

        elapsed = time.time() - start
        themes = pred.get("themes", [])
        match = row["primary"] in themes
        results.append({
            "url": row["url"],
            "true_primary": row["primary"],
            "pred_themes": themes,
            "match": match,
        })
        symbol = "✓" if match else "✗"
        acc = sum(r["match"] for r in results) / len(results)
        print(f"  {symbol} [{i+1}/{total}] {elapsed:.1f}s acc={acc:.0%} pred={themes}", flush=True)

    print(f"\nFinished in {time.time()-run_start:.0f}s", flush=True)


    # Summary
    print("\n" + "=" * 60, flush=True)
    acc = sum(r["match"] for r in results) / len(results)
    print(f"ACCURACY: {acc:.0%} ({sum(r['match'] for r in results)}/{len(results)})")

    # Per-theme breakdown
    res_df = pd.DataFrame(results)
    print(f"\nPer-theme accuracy:")
    for theme, group in res_df.groupby("true_primary"):
        theme_acc = group["match"].mean()
        print(f"  {theme:<35} {theme_acc:.0%} ({group['match'].sum()}/{len(group)})")

    # Save full results
    res_df.to_csv("eval_results.csv", index=False)
    print(f"\nFull results saved to eval_results.csv")


if __name__ == "__main__":
    asyncio.run(main())
