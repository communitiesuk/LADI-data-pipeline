"""
Stratified 50k input selection for LADI pipeline.

Guarantees:
  - At least MIN_IFS Infrastructure Funding Statement docs (detected by URL/text)
  - At least MIN_DEPTH_LAs local authorities with 500+ docs each (depth coverage)
  - Broad coverage across remaining LAs

Two-pass to avoid OOM:
  Pass 1 — metadata only (URL, LA, is_IFS flag). Builds selection set.
  Pass 2 — streams full rows for selected URLs to output CSV.

Usage:
    python scripts/prepare_input_stratified.py
    python scripts/prepare_input_stratified.py --n 50000 --output data/summarise_input_50k.csv
"""
import argparse
import logging
import re
from collections import defaultdict
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

IFS_URL_PATTERNS = [
    re.compile(r'infrastructure.funding.statement', re.I),
    re.compile(r'infrastructure.funding', re.I),
    re.compile(r'[\-_/]ifs[\-_.]', re.I),
    re.compile(r'[\-_/]ifs$', re.I),
    re.compile(r'annual.ifs', re.I),
]
IFS_TEXT_PATTERN = re.compile(r'infrastructure funding statement', re.I)


def is_ifs(url: str, text_prefix: str) -> bool:
    for pat in IFS_URL_PATTERNS:
        if pat.search(url):
            return True
    return bool(IFS_TEXT_PATTERN.search(text_prefix))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=50_000)
    p.add_argument('--output', default=None)
    p.add_argument('--input', default=CRAWL_CSV)
    p.add_argument('--min-ifs', type=int, default=50)
    p.add_argument('--min-depth-las', type=int, default=10)
    p.add_argument('--depth-threshold', type=int, default=500)
    args = p.parse_args()

    target = args.n
    output = args.output or f'data/summarise_input_{target // 1000}k.csv'
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f'Target: {target:,} | min IFS: {args.min_ifs} | '
             f'min depth LAs: {args.min_depth_las} (≥{args.depth_threshold} docs each)')

    # ------------------------------------------------------------------ PASS 1
    # Collect metadata only: URL → {la, is_ifs}
    # Per-LA ordered URL lists for later selection
    log.info('Pass 1: scanning for metadata ...')

    ifs_urls: list[str] = []          # IFS docs in discovery order
    la_urls: dict[str, list[str]] = defaultdict(list)  # non-IFS, per LA
    url_la: dict[str, str] = {}       # url → authority (for pass 2 stats)
    seen: set[str] = set()
    total_scanned = 0

    for chunk in pd.read_csv(
        args.input,
        usecols=['Authority Name', 'Document Link', 'text', 'extraction_status'],
        chunksize=CHUNK_SIZE,
        low_memory=False,
        on_bad_lines='skip',
    ):
        total_scanned += len(chunk)

        chunk = chunk.dropna(subset=['Document Link', 'text'])
        chunk['text'] = chunk['text'].astype(str)
        chunk = chunk[chunk['text'].str.split().str.len() >= MIN_WORDS]
        chunk = chunk[chunk['extraction_status'].fillna('').str.lower() != 'failed']
        chunk = chunk[~chunk['Document Link'].isin(seen)]
        chunk = chunk.drop_duplicates(subset=['Document Link'])
        seen.update(chunk['Document Link'].tolist())

        for _, row in chunk.iterrows():
            url = str(row['Document Link'])
            la = str(row.get('Authority Name', 'Unknown'))
            text_prefix = str(row['text'])[:500]

            url_la[url] = la
            if is_ifs(url, text_prefix):
                ifs_urls.append(url)
            else:
                la_urls[la].append(url)

        if total_scanned % 200_000 == 0:
            log.info(f'  Scanned {total_scanned:,} | IFS: {len(ifs_urls)} | '
                     f'LAs: {len(la_urls)} | qualified unique: {len(url_la)}')

    log.info(f'Pass 1 done — {total_scanned:,} rows scanned')
    log.info(f'  Qualified unique docs: {len(url_la):,}')
    log.info(f'  IFS docs found: {len(ifs_urls)}')
    log.info(f'  LAs found: {len(la_urls)}')

    # ----------------------------------------------------------------- SELECT
    selected: set[str] = set()

    def add_urls(urls):
        for u in urls:
            if len(selected) >= target:
                break
            selected.add(u)

    # 1. IFS docs first
    add_urls(ifs_urls)
    log.info(f'After IFS: {len(selected):,} selected')

    # 2. Depth LAs — pick top N LAs by doc count, take exactly DEPTH_PER_LA from each
    #    so they are guaranteed to appear 500+ times in the output.
    la_sizes = sorted(la_urls.items(), key=lambda x: -len(x[1]))
    depth_las = [(la, urls) for la, urls in la_sizes if len(urls) >= args.depth_threshold]
    log.info(f'LAs with {args.depth_threshold}+ docs in corpus: {len(depth_las)}')

    # Take the top min_depth_las LAs and reserve depth_per_la slots each
    depth_per_la = 1500
    chosen_depth_las = depth_las[:args.min_depth_las]
    for la, urls in chosen_depth_las:
        log.info(f'  {la}: {len(urls)} corpus docs → taking {min(len(urls), depth_per_la)}')
        add_urls(urls[:depth_per_la])
    log.info(f'After depth LAs ({len(chosen_depth_las)}): {len(selected):,} selected')

    # 3. Fill remaining from ALL other LAs — round-robin for breadth
    chosen_depth_names = {la for la, _ in chosen_depth_las}
    other_las = [(la, urls) for la, urls in la_sizes
                 if la not in chosen_depth_names]
    remaining = target - len(selected)
    if remaining > 0 and other_las:
        quota = max(1, remaining // len(other_las))
        log.info(f'Filling {remaining:,} slots from {len(other_las)} other LAs (~{quota}/LA)')
        for _, urls in other_las:
            add_urls(urls[:quota])
        # Second pass for any shortfall
        if len(selected) < target:
            for _, urls in other_las:
                add_urls(urls[quota:])
                if len(selected) >= target:
                    break

    log.info(f'Final selection: {len(selected):,} URLs')

    # ------------------------------------------------------------------ PASS 2
    # Stream full rows for selected URLs to output CSV
    log.info('Pass 2: writing selected rows ...')

    written = 0
    first_chunk = True

    for chunk in pd.read_csv(
        args.input,
        usecols=['Authority Name', 'Document Link', 'text'],
        chunksize=CHUNK_SIZE,
        low_memory=False,
        on_bad_lines='skip',
    ):
        keep = chunk[chunk['Document Link'].isin(selected)]
        if len(keep) == 0:
            continue
        keep.to_csv(
            output_path,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            index=False,
        )
        first_chunk = False
        written += len(keep)

        if written % 10_000 == 0 or written == len(selected):
            log.info(f'  Written {written:,}/{len(selected):,}')

    log.info(f'Pass 2 done — {written:,} rows written to {output_path}')

    # ------------------------------------------------------------------ STATS
    result = pd.read_csv(output_path)
    ifs_count = sum(
        1 for _, r in result.iterrows()
        if is_ifs(str(r['Document Link']), str(r['text'])[:500])
    )
    auth_counts = result['Authority Name'].value_counts()
    deep_count = sum(1 for c in auth_counts if c >= args.depth_threshold)

    log.info('--- Summary ---')
    log.info(f'IFS docs:               {ifs_count} (target ≥{args.min_ifs})')
    log.info(f'Unique LAs:             {result["Authority Name"].nunique()}')
    log.info(f'LAs with {args.depth_threshold}+ docs:      {deep_count} (target ≥{args.min_depth_las})')
    log.info(f'Top 10 LAs by doc count:')
    for la, cnt in auth_counts.head(10).items():
        log.info(f'  {la}: {cnt}')

    if ifs_count < args.min_ifs:
        log.warning(f'Only {ifs_count} IFS docs — below target {args.min_ifs}')
    if deep_count < args.min_depth_las:
        log.warning(f'Only {deep_count} depth LAs — below target {args.min_depth_las}')


if __name__ == '__main__':
    main()
