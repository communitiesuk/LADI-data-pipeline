"""Evaluate the summarise stage against manually-annotated gold standard docs."""
import asyncio
import importlib.util
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# Import call_api from stages/02_summarise.py
_spec = importlib.util.spec_from_file_location(
    "summarise", Path(__file__).parent / "stages" / "02_summarise.py"
)
_summarise = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_summarise)
call_api = _summarise.call_api
clean_text = _summarise.clean_text
truncate_words = _summarise.truncate_words
THEMES = _summarise.THEMES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GOLD_STANDARD = "eval_gold_standard.csv"
OUTPUT_FILE = "eval_summarise_results.csv"

# IFS detection: primary=Planning and development + secondary_1=Local government finance
IFS_PRIMARY = "Planning and development"
IFS_SECONDARY = "Local government finance"

# Thresholds
GENERAL_THRESHOLDS = {
    "themes_primary": 0.75,
    "year_off_by_one": 0.75,
    "title_mean": 3.5,
    "summary_mean": 3.5,
}
IFS_THRESHOLDS = {
    "themes_primary": 0.85,
    "year_off_by_one": 0.80,
    "title_mean": 4.0,
    "summary_mean": 4.0,
}

# ---------------------------------------------------------------------------
# LLM Judge prompts
# ---------------------------------------------------------------------------

TITLE_JUDGE_PROMPT = """You are evaluating whether a predicted document title matches the true title.

True title: {gold_title}
Predicted title: {pred_title}

Document text (first 200 words): {text_snippet}

Score the predicted title from 1-5:
- 5: Accurate and specific — clearly refers to the same document
- 4: Correct document, minor wording differences
- 3: Reasonable but vague or partially correct
- 2: Misleading or too generic
- 1: Wrong — refers to a different document or is nonsensical

Respond with JSON: {{"score": <int>, "reason": "<brief explanation>"}}"""

SUMMARY_JUDGE_PROMPT = """You are evaluating the quality of a document summary.

Source document text (first 500 words):
{source_text}

Predicted summary:
{pred_summary}

Score the summary from 1-5:
- 5: Faithful, specific, and useful for search — captures the document's purpose and key content
- 4: Accurate and useful, minor omissions
- 3: Correct main point but thin or generic
- 2: Partially correct but misleading or missing key content
- 1: Hallucinated, factually wrong, or completely off-topic

Respond with JSON: {{"score": <int>, "reason": "<brief explanation>"}}"""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def build_client() -> AsyncOpenAI:
    from ladi.apim import _get_token
    return AsyncOpenAI(
        base_url=os.environ["LADI_APIM_BASE_URL"],
        api_key=_get_token(),
        default_headers={"Ocp-Apim-Subscription-Key": os.environ["LADI_APIM_SUBSCRIPTION_KEY"]},
        default_query={"api-version": os.environ["LADI_APIM_API_VERSION"]},
        max_retries=0,
        timeout=30.0,
    )


# ---------------------------------------------------------------------------
# Judge functions
# ---------------------------------------------------------------------------

async def judge_call(client: AsyncOpenAI, prompt: str) -> dict:
    """Make a judge LLM call, return parsed JSON response."""
    resp = await client.chat.completions.create(
        model="gpt-4-1",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=150,
    )
    return json.loads(resp.choices[0].message.content)


async def judge_title(client: AsyncOpenAI, gold_title: str, pred_title: str, text: str) -> dict:
    snippet = " ".join(clean_text(text).split()[:200])
    prompt = TITLE_JUDGE_PROMPT.format(
        gold_title=gold_title, pred_title=pred_title, text_snippet=snippet
    )
    return await judge_call(client, prompt)


async def judge_summary(client: AsyncOpenAI, source_text: str, pred_summary: str) -> dict:
    snippet = " ".join(clean_text(source_text).split()[:500])
    # WAF blocks "personal data:" followed by PII field names — swap colon for dash
    snippet = snippet.replace("personal data:", "personal data -")
    prompt = SUMMARY_JUDGE_PROMPT.format(source_text=snippet, pred_summary=pred_summary)
    return await judge_call(client, prompt)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_themes(gold_row: pd.Series, pred: dict) -> dict:
    """Score theme prediction against gold standard."""
    pred_themes = pred.get("themes", [])
    gold_themes = set()
    for col in ["primary", "secondary_1", "secondary_2"]:
        val = gold_row.get(col)
        if pd.notna(val) and str(val).strip():
            gold_themes.add(str(val).strip())

    primary = str(gold_row["primary"]).strip()
    primary_match = primary in pred_themes
    any_match = bool(gold_themes & set(pred_themes))
    spurious = [t for t in pred_themes if t not in gold_themes]

    return {
        "primary_match": primary_match,
        "any_match": any_match,
        "spurious_themes": spurious,
    }


def score_year(gold_year, pred_year) -> dict:
    """Score year prediction."""
    # Normalise
    gold_y = None if pd.isna(gold_year) else int(gold_year)
    pred_y = None if pred_year is None else int(pred_year) if pred_year else None

    if gold_y is None and pred_y is None:
        return {"exact_match": True, "off_by_one": True, "null_miss": False}
    if gold_y is None or pred_y is None:
        return {"exact_match": False, "off_by_one": False, "null_miss": True}

    exact = gold_y == pred_y
    off_by_one = abs(gold_y - pred_y) <= 1
    return {"exact_match": exact, "off_by_one": off_by_one, "null_miss": False}


# ---------------------------------------------------------------------------
# Main evaluation loop (concurrent with shared rate-limit pause)
# ---------------------------------------------------------------------------

CONCURRENCY = 3

# Global pause event: when cleared, all workers block before their next API call.
# Only the worker that hits 429 clears it, sleeps, then sets it again.
_pause_event: asyncio.Event | None = None
_pause_lock: asyncio.Lock | None = None


def _init_pause():
    """Lazily create the event+lock (must be called inside a running event loop)."""
    global _pause_event, _pause_lock
    if _pause_event is None:
        _pause_event = asyncio.Event()
        _pause_event.set()  # start unblocked
        _pause_lock = asyncio.Lock()


async def rate_limited_call(coro_fn, *args, **kwargs):
    """Call an async function; if 429, freeze ALL workers for 65s then retry."""
    _init_pause()
    # Wait if another worker triggered a pause
    await _pause_event.wait()
    try:
        return await coro_fn(*args, **kwargs)
    except Exception as e:
        if "429" in str(e):
            # Only one worker performs the pause
            async with _pause_lock:
                if _pause_event.is_set():
                    _pause_event.clear()
                    print(f"  ⏳ Rate limited — all workers pausing 65s...", flush=True)
                    await asyncio.sleep(65)
                    _pause_event.set()
            # Wait again in case we weren't the one who did the sleep
            await _pause_event.wait()
            return await coro_fn(*args, **kwargs)
        raise


async def evaluate_one(client: AsyncOpenAI, row: pd.Series, idx: int, total: int) -> dict | None:
    """Evaluate a single document: summarise + judge title + judge summary."""
    start = time.time()
    url = row["url"]
    authority = str(row.get("authority", ""))
    text = str(row["text"])

    # Step 1: Get summarise prediction
    try:
        pred = await rate_limited_call(call_api, client, url, authority, text)
    except Exception as e:
        print(f"  ERROR [{idx+1}/{total}] {type(e).__name__}: {str(e)[:80]}", flush=True)
        return None

    # Step 2: Score themes and year (automatic — no API call)
    theme_scores = score_themes(row, pred)
    year_scores = score_year(row.get("year"), pred.get("year"))

    # Step 3: Judge title
    try:
        title_result = await rate_limited_call(
            judge_title, client, str(row["title"]), pred.get("title", ""), text
        )
    except Exception as e:
        title_result = {"score": 0, "reason": f"judge error: {e}"}

    # Step 4: Judge summary
    try:
        summary_result = await rate_limited_call(
            judge_summary, client, text, pred.get("summary", "")
        )
    except Exception as e:
        summary_result = {"score": 0, "reason": f"judge error: {e}"}

    elapsed = time.time() - start

    result = {
        "url": url,
        "authority": authority,
        "gold_primary": row["primary"],
        "gold_title": row["title"],
        "gold_year": row.get("year"),
        "pred_title": pred.get("title", ""),
        "pred_year": pred.get("year"),
        "pred_summary": pred.get("summary", ""),
        "pred_themes": pred.get("themes", []),
        "primary_match": theme_scores["primary_match"],
        "any_match": theme_scores["any_match"],
        "spurious_themes": theme_scores["spurious_themes"],
        "year_exact": year_scores["exact_match"],
        "year_off_by_one": year_scores["off_by_one"],
        "year_null_miss": year_scores["null_miss"],
        "title_score": title_result.get("score", 0),
        "title_reason": title_result.get("reason", ""),
        "summary_score": summary_result.get("score", 0),
        "summary_reason": summary_result.get("reason", ""),
        "is_ifs": (str(row["primary"]).strip() == IFS_PRIMARY and
                   str(row.get("secondary_1", "")).strip() == IFS_SECONDARY),
    }

    # Progress
    symbol = "✓" if theme_scores["primary_match"] else "✗"
    print(
        f"  {symbol} [{idx+1}/{total}] {elapsed:.1f}s "
        f"theme={'✓' if theme_scores['primary_match'] else '✗'} "
        f"year={'✓' if year_scores['off_by_one'] else '✗'} "
        f"title={title_result.get('score', '?')}/5 "
        f"summary={summary_result.get('score', '?')}/5 "
        f"pred_themes={pred.get('themes', [])}",
        flush=True,
    )
    return result


async def main():
    run_start = time.time()

    df = pd.read_csv(GOLD_STANDARD)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.split().str.len() >= 50]
    total = len(df)
    print(f"Evaluating {total} documents (concurrency={CONCURRENCY})\n", flush=True)

    client = build_client()

    # Process with limited concurrency via semaphore
    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded_evaluate(row, idx):
        async with sem:
            return await evaluate_one(client, row, idx, total)

    tasks = [bounded_evaluate(row, i) for i, (_, row) in enumerate(df.iterrows())]
    raw_results = await asyncio.gather(*tasks)
    results = [r for r in raw_results if r is not None]

    # =========================================================================
    # REPORT
    # =========================================================================
    print(f"\nFinished in {time.time()-run_start:.0f}s")
    print("\n" + "=" * 70)

    res_df = pd.DataFrame(results)
    n = len(res_df)

    # -- Overall metrics --
    theme_primary_pct = res_df["primary_match"].mean()
    theme_any_pct = res_df["any_match"].mean()
    spurious_rate = res_df["spurious_themes"].apply(len).gt(0).mean()
    year_exact_pct = res_df["year_exact"].mean()
    year_obo_pct = res_df["year_off_by_one"].mean()
    year_null_misses = res_df["year_null_miss"].sum()
    title_mean = res_df["title_score"].mean()
    title_median = res_df["title_score"].median()
    title_fail = res_df["title_score"].lt(3).mean()
    summary_mean = res_df["summary_score"].mean()
    summary_median = res_df["summary_score"].median()
    summary_fail = res_df["summary_score"].lt(3).mean()

    print(f"\nOVERALL (n={n}):")
    print(f"  THEMES:   primary_match={theme_primary_pct:.0%}  any_match={theme_any_pct:.0%}  spurious_rate={spurious_rate:.0%}")
    print(f"  YEAR:     exact={year_exact_pct:.0%}  off_by_one={year_obo_pct:.0%}  null_misses={year_null_misses}")
    print(f"  TITLE:    mean={title_mean:.1f}  median={title_median:.0f}  fail_rate={title_fail:.0%} (<3)")
    print(f"  SUMMARY:  mean={summary_mean:.1f}  median={summary_median:.0f}  fail_rate={summary_fail:.0%} (<3)")

    # -- IFS-specific --
    ifs_df = res_df[res_df["is_ifs"]]
    if len(ifs_df) > 0:
        print(f"\nIFS DOCS (n={len(ifs_df)}):")
        print(f"  THEMES:   primary_match={ifs_df['primary_match'].mean():.0%}")
        print(f"  YEAR:     off_by_one={ifs_df['year_off_by_one'].mean():.0%}")
        print(f"  TITLE:    mean={ifs_df['title_score'].mean():.1f}")
        print(f"  SUMMARY:  mean={ifs_df['summary_score'].mean():.1f}")

    # -- Per-theme breakdown --
    print(f"\nPer-theme breakdown:")
    for theme, group in res_df.groupby("gold_primary"):
        pct = group["primary_match"].mean()
        print(f"  {theme:<35} {group['primary_match'].sum()}/{len(group)}  {pct:.0%}")

    # -- Failures --
    failures = res_df[(res_df["title_score"] < 3) | (res_df["summary_score"] < 3)]
    if len(failures) > 0:
        print(f"\nFailures (title or summary < 3):")
        for _, f in failures.iterrows():
            print(f"  [{f['url'][:70]}...]")
            if f["title_score"] < 3:
                print(f"    title={f['title_score']}/5: {f['title_reason']}")
            if f["summary_score"] < 3:
                print(f"    summary={f['summary_score']}/5: {f['summary_reason']}")

    # -- Pass/Fail verdict --
    print("\n" + "=" * 70)
    general_pass = (
        theme_primary_pct >= GENERAL_THRESHOLDS["themes_primary"]
        and year_obo_pct >= GENERAL_THRESHOLDS["year_off_by_one"]
        and title_mean >= GENERAL_THRESHOLDS["title_mean"]
        and summary_mean >= GENERAL_THRESHOLDS["summary_mean"]
    )
    print(f"GENERAL: {'PASS ✓' if general_pass else 'FAIL ✗'}")
    print(f"  themes≥{GENERAL_THRESHOLDS['themes_primary']:.0%}: {theme_primary_pct:.0%} {'✓' if theme_primary_pct >= GENERAL_THRESHOLDS['themes_primary'] else '✗'}")
    print(f"  year≥{GENERAL_THRESHOLDS['year_off_by_one']:.0%}: {year_obo_pct:.0%} {'✓' if year_obo_pct >= GENERAL_THRESHOLDS['year_off_by_one'] else '✗'}")
    print(f"  title≥{GENERAL_THRESHOLDS['title_mean']}: {title_mean:.1f} {'✓' if title_mean >= GENERAL_THRESHOLDS['title_mean'] else '✗'}")
    print(f"  summary≥{GENERAL_THRESHOLDS['summary_mean']}: {summary_mean:.1f} {'✓' if summary_mean >= GENERAL_THRESHOLDS['summary_mean'] else '✗'}")

    if len(ifs_df) > 0:
        ifs_pass = (
            ifs_df["primary_match"].mean() >= IFS_THRESHOLDS["themes_primary"]
            and ifs_df["year_off_by_one"].mean() >= IFS_THRESHOLDS["year_off_by_one"]
            and ifs_df["title_score"].mean() >= IFS_THRESHOLDS["title_mean"]
            and ifs_df["summary_score"].mean() >= IFS_THRESHOLDS["summary_mean"]
        )
        print(f"\nIFS:     {'PASS ✓' if ifs_pass else 'FAIL ✗'}")
        print(f"  themes≥{IFS_THRESHOLDS['themes_primary']:.0%}: {ifs_df['primary_match'].mean():.0%} {'✓' if ifs_df['primary_match'].mean() >= IFS_THRESHOLDS['themes_primary'] else '✗'}")
        print(f"  year≥{IFS_THRESHOLDS['year_off_by_one']:.0%}: {ifs_df['year_off_by_one'].mean():.0%} {'✓' if ifs_df['year_off_by_one'].mean() >= IFS_THRESHOLDS['year_off_by_one'] else '✗'}")
        print(f"  title≥{IFS_THRESHOLDS['title_mean']}: {ifs_df['title_score'].mean():.1f} {'✓' if ifs_df['title_score'].mean() >= IFS_THRESHOLDS['title_mean'] else '✗'}")
        print(f"  summary≥{IFS_THRESHOLDS['summary_mean']}: {ifs_df['summary_score'].mean():.1f} {'✓' if ifs_df['summary_score'].mean() >= IFS_THRESHOLDS['summary_mean'] else '✗'}")

    # Save results
    res_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFull results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
