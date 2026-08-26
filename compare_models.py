"""
Compare GPT-4.1 vs GPT-5-mini on N already-summarised documents.

Uses embeddings_combined_08072026.csv which has both the original text
(text column) and the GPT-4.1 outputs (title, year, summary, themes).

Usage:
    python compare_models.py
    python compare_models.py --n 20
    python compare_models.py --input data/embeddings_combined_08072026.csv
"""
import argparse
import asyncio
import importlib.util
import json
import re
import sys
import textwrap
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# stages/02_summarise.py starts with a digit — must use importlib
_spec = importlib.util.spec_from_file_location(
    "summarise",
    Path(__file__).parent / "stages" / "02_summarise.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_client = _mod.build_client
clean_text = _mod.clean_text
truncate_words = _mod.truncate_words
MODEL_PROFILES = _mod.MODEL_PROFILES
SYSTEM_PROMPT = _mod.SYSTEM_PROMPT
async def run_model(rows: list[dict], profile: dict, delay_s: float = 8.0) -> list[dict]:
    client = build_client(profile)
    results = []

    for i, row in enumerate(rows):
        if i > 0:
            await asyncio.sleep(delay_s)

        url = row['url']
        authority = row.get('authority', '')
        text = str(row.get('text', ''))

        try:
            cleaned = clean_text(truncate_words(text))
            cleaned = cleaned.replace(":", " -")
            cleaned = re.sub(r"https?://\S+", "", cleaned)
            cleaned = re.sub(r"www\.\S+", "", cleaned)
            cleaned = re.sub(r"[\w.-]+@[\w.-]+\.\w+", "", cleaned)
            cleaned = re.sub(r"\d{4,5}\s?\d{6}", "", cleaned)
            cleaned = re.sub(r"[^\x00-\x7F]+", "", cleaned)

            alpha_ratio = sum(c.isalpha() for c in cleaned) / max(len(cleaned), 1)
            if alpha_ratio < 0.5:
                print(f"  [SKIP garbled] {url[:70]}")
                continue

            user_content = f"Authority: {authority}\nURL: {url}\n\n{cleaned}"
            create_kwargs = dict(
                model=profile['deployment_id'],
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_content},
                ],
                response_format={'type': 'json_object'},
            )
            if profile['is_reasoning']:
                create_kwargs['max_completion_tokens'] = profile['max_output_tokens']
            else:
                create_kwargs['temperature'] = 0.0
                create_kwargs['max_tokens'] = profile['max_output_tokens']
            resp = await client.chat.completions.create(**create_kwargs)
            output = json.loads(resp.choices[0].message.content)
            usage = resp.usage
            results.append({
                'url': url,
                'authority': authority,
                'ref_title':   row.get('title'),
                'ref_year':    row.get('year'),
                'ref_themes':  row.get('themes'),
                'ref_summary': row.get('summary'),
                'new_title':   output.get('title'),
                'new_year':    output.get('year'),
                'new_themes':  output.get('themes'),
                'new_summary': output.get('summary'),
                'tokens_in':    usage.prompt_tokens if usage else None,
                'tokens_out':   usage.completion_tokens if usage else None,
                'tokens_total': usage.total_tokens if usage else None,
            })
            print(f"  [OK] {url[:65]}  in={usage.prompt_tokens} out={usage.completion_tokens}")

        except Exception as e:
            print(f"  [ERR] {url[:70]}: {e}")
            results.append({'url': url, 'authority': authority, 'error': str(e)})

    return results


def print_comparison(results: list[dict]) -> None:
    sep = "=" * 80
    thin = "-" * 80
    total_in = total_out = 0
    ok_count = 0

    for i, r in enumerate(results, 1):
        if 'error' in r:
            print(f"\n{sep}\nDoc {i}: ERROR — {r['error']}\nURL: {r['url']}\n")
            continue

        ok_count += 1
        total_in += r.get('tokens_in') or 0
        total_out += r.get('tokens_out') or 0

        print(f"\n{sep}")
        print(f"Doc {i}: {r['url'][:75]}")
        print(f"Authority: {r['authority']}")
        print(f"Tokens: {r['tokens_in']} in / {r['tokens_out']} out (total {r['tokens_total']})")
        print(thin)

        print("TITLE")
        print(f"  Ref (4.1): {r['ref_title']}")
        print(f"  New model: {r['new_title']}")

        print("YEAR")
        print(f"  Ref (4.1): {r['ref_year']}")
        print(f"  New model: {r['new_year']}")

        print("THEMES")
        print(f"  Ref (4.1): {r['ref_themes']}")
        print(f"  New model: {r['new_themes']}")

        print("SUMMARY")
        ref = textwrap.fill(
            str(r['ref_summary'] or ''), width=76,
            initial_indent='  Ref (4.1): ', subsequent_indent='            ',
        )
        new = textwrap.fill(
            str(r['new_summary'] or ''), width=76,
            initial_indent='  New model: ', subsequent_indent='            ',
        )
        print(ref)
        print(new)

    print(f"\n{sep}")
    print(f"TOTALS — {ok_count} docs successfully compared")
    if ok_count:
        print(f"  Input tokens:  {total_in:,}  (avg {total_in // ok_count:,}/doc)")
        print(f"  Output tokens: {total_out:,}  (avg {total_out // ok_count:,}/doc)")
        print(f"  Total tokens:  {total_in + total_out:,}")
    print(sep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/embeddings_combined_08072026.csv',
                   help='CSV with text + GPT-4.1 outputs (default: embeddings_combined_08072026.csv)')
    p.add_argument('--n', type=int, default=10, help='Number of docs to test (default: 10)')
    p.add_argument('--model', default='gpt5.1', choices=list(MODEL_PROFILES),
                   help='Model profile to test against the 4.1 reference (default: gpt5.1)')
    p.add_argument('--delay', type=float, default=8.0,
                   help='Seconds between calls to avoid rate limits (default: 8)')
    args = p.parse_args()

    profile = MODEL_PROFILES[args.model]
    print(f"Model: {args.model}  deployment={profile['deployment_id']}  reasoning={profile['is_reasoning']}")
    print(f"Loading {args.n} rows from {args.input} ...")
    df = pd.read_csv(
        args.input,
        usecols=['url', 'authority', 'title', 'year', 'summary', 'themes', 'text'],
        nrows=args.n,
    )
    df = df.dropna(subset=['text', 'summary'])
    rows = df.to_dict('records')
    print(f"Loaded {len(rows)} rows with text and summary\n")
    print(f"Running {args.model} on {len(rows)} docs ...")

    results = asyncio.run(run_model(rows, profile, delay_s=args.delay))
    print_comparison(results)


if __name__ == '__main__':
    main()
