"""
Extract N filtered rows from the main crawl CSV for use as pipeline input.

Filters: ≥100 words, non-null text, successful extraction, deduped by URL.
Reads the 24GB CSV in chunks so it doesn't blow memory.

Usage:
    python scripts/prepare_input.py --n 50000 --output data/summarise_input_50k.csv
    python scripts/prepare_input.py --n 10000 --output data/summarise_input_10k_fresh.csv
"""
import argparse
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

CRAWL_CSV = 'found_links_hybrid_20260430_154001_text.csv'
MIN_WORDS = 100
CHUNK_SIZE = 10_000


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=50_000, help='Number of docs to extract (default: 50000)')
    p.add_argument('--output', default=None, help='Output CSV path (default: data/summarise_input_{n}.csv)')
    p.add_argument('--input', default=CRAWL_CSV, help=f'Source crawl CSV (default: {CRAWL_CSV})')
    p.add_argument('--skip', type=int, default=0, help='Skip first N matching rows (for pagination)')
    args = p.parse_args()

    output = args.output or f'data/summarise_input_{args.n // 1000}k.csv'
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        log.warning(f'{output_path} already exists — will overwrite')

    log.info(f'Reading {args.input} in chunks of {CHUNK_SIZE:,} ...')
    log.info(f'Target: {args.n:,} docs with ≥{MIN_WORDS} words | output: {output_path}')

    seen_urls: set = set()
    collected: list[pd.DataFrame] = []
    total_collected = 0
    total_scanned = 0
    skipped = 0

    for chunk in pd.read_csv(
        args.input,
        usecols=['Authority Name', 'Document Link', 'text', 'extraction_status'],
        chunksize=CHUNK_SIZE,
        low_memory=False,
        on_bad_lines='skip',
    ):
        total_scanned += len(chunk)

        # Basic filters
        chunk = chunk.dropna(subset=['Document Link', 'text'])
        chunk['text'] = chunk['text'].astype(str)
        chunk = chunk[chunk['text'].str.split().str.len() >= MIN_WORDS]
        chunk = chunk[chunk['extraction_status'].fillna('').str.lower() != 'failed']

        # Deduplicate by URL
        chunk = chunk[~chunk['Document Link'].isin(seen_urls)]
        chunk = chunk.drop_duplicates(subset=['Document Link'])
        seen_urls.update(chunk['Document Link'].tolist())

        if skipped < args.skip:
            to_skip = min(len(chunk), args.skip - skipped)
            chunk = chunk.iloc[to_skip:]
            skipped += to_skip

        remaining_needed = args.n - total_collected
        if len(chunk) > remaining_needed:
            chunk = chunk.iloc[:remaining_needed]

        if len(chunk) > 0:
            collected.append(chunk)
            total_collected += len(chunk)

        if total_scanned % 100_000 == 0:
            log.info(f'Scanned {total_scanned:,} rows | collected {total_collected:,}/{args.n:,}')

        if total_collected >= args.n:
            break

    log.info(f'Scanned {total_scanned:,} total rows | collected {total_collected:,}')

    if not collected:
        log.error('No rows collected — check input file and filters')
        return

    result = pd.concat(collected, ignore_index=True)
    result = result.rename(columns={'Document Link': 'Document Link'})
    result.to_csv(output_path, index=False)
    log.info(f'Written {len(result):,} rows to {output_path}')

    # Summary stats
    auth_counts = result['Authority Name'].value_counts()
    log.info(f'Unique authorities: {result["Authority Name"].nunique():,}')
    log.info(f'Top 5 authorities: {dict(auth_counts.head())}')


if __name__ == '__main__':
    main()
