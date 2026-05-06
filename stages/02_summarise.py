"""
Stage 2: Summarise and classify documents using GPT via Azure APIM.

For each document with ≥100 words, calls GPT to extract:
  title, year, summary, themes, document_type, confidence

Output: JSONL (one JSON object per line, appended incrementally).
Supports checkpoint/resume — already-processed URLs are skipped on restart.
"""
import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import yaml
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

THEMES = [
    'finance', 'planning', 'housing', 'transport', 'environment',
    'education', 'health', 'social_care', 'children_services',
    'economic_development', 'governance', 'waste', 'leisure', 'digital', 'other',
]

DOC_TYPES = [
    'Local Plan', 'Infrastructure Funding Statement', 'Medium Term Financial Strategy',
    'Annual Report', 'Statement of Accounts', 'Budget', 'Corporate Plan',
    'Sustainability Appraisal', 'Housing Needs Assessment', 'Transport Plan',
    'Climate Action Plan', 'Equality Impact Assessment', 'Asset Management Plan',
    'Community Infrastructure Levy', 'Neighbourhood Plan', 'other',
]

SYSTEM_PROMPT = (
    "You are a metadata extractor for UK local authority documents.\n"
    "Given document text, return JSON with exactly these fields:\n"
    "- title: string (document title; infer from content if not stated)\n"
    "- year: integer or null (publication year; null if unknown)\n"
    "- summary: string (2-3 sentences describing purpose and key content)\n"
    f"- themes: array of 1-3 strings from: {', '.join(THEMES)}\n"
    f"- document_type: string, one of: {', '.join(DOC_TYPES)}\n"
    "- confidence: float 0.0-1.0 (how confident you are in the classification)\n\n"
    "Respond with valid JSON only. No markdown fences."
)

# CSV column names from crawldocs output
URL_COL = 'Document Link'
TEXT_COL = 'text'
AUTH_COL = 'Authority Name'


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate to approximately max_chars, cutting at a word boundary."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    boundary = cut.rfind(' ')
    return cut[:boundary] if boundary > max_chars * 0.8 else cut


def load_checkpoint(output_path: Path) -> set:
    """Return set of URLs already present in the output JSONL."""
    done: set = set()
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if 'url' in obj:
                    done.add(obj['url'])
            except json.JSONDecodeError:
                pass  # skip malformed lines (e.g. truncated on crash)
    if done:
        logger.info(f"Checkpoint: {len(done):,} already processed — resuming")
    return done


async def call_api(
    client: AsyncOpenAI,
    model: str,
    url: str,
    authority: str,
    text: str,
    max_chars: int,
) -> dict:
    truncated = truncate_text(text, max_chars)
    user_content = f"Authority: {authority}\nURL: {url}\n\n{truncated}"

    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_content},
        ],
        response_format={'type': 'json_object'},
        temperature=0.0,
        max_tokens=400,
    )
    result = json.loads(resp.choices[0].message.content)
    result.update({'url': url, 'authority': authority, 'model': model})
    return result


async def worker(
    queue: asyncio.Queue,
    client: AsyncOpenAI,
    models: list,
    model_idx: list,  # single-element list used as mutable counter
    max_chars: int,
    out_fh,
    counters: dict,
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        url, authority, text = item
        model = models[model_idx[0] % len(models)]
        model_idx[0] += 1

        try:
            result = await call_api(client, model, url, authority, text, max_chars)
            out_fh.write(json.dumps(result) + '\n')
            out_fh.flush()
            counters['done'] += 1
        except Exception as e:
            counters['errors'] += 1
            logger.warning(f"[{url[:80]}] {type(e).__name__}: {e}")

        queue.task_done()


async def stats_loop(counters: dict, total: int, interval: int = 60) -> None:
    start = time.monotonic()
    try:
        while True:
            await asyncio.sleep(interval)
            elapsed = time.monotonic() - start
            done = counters['done']
            errors = counters['errors']
            rate = done / elapsed * 60 if elapsed > 0 else 0
            remaining = total - done - errors
            eta = f"{remaining / rate:.0f} min" if rate > 0 else "unknown"
            logger.info(
                f"Progress: {done:,}/{total:,} done | {errors:,} errors | "
                f"{rate:.0f} docs/min | ETA {eta}"
            )
    except asyncio.CancelledError:
        pass


async def run(cfg: dict, input_csv: str, output_file: str, concurrency: int) -> None:
    sum_cfg = cfg['summarise']
    apim_cfg = cfg['apim']

    models: list = sum_cfg.get('fallback_models') or [sum_cfg['model']]
    # ~4 chars per token for a rough but dependency-free truncation
    max_chars: int = sum_cfg.get('max_input_tokens', 1500) * 4
    min_words: int = sum_cfg.get('min_words', 100)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get('APIM_API_KEY') or apim_cfg.get('api_key') or 'placeholder'
    endpoint = os.environ.get('APIM_ENDPOINT') or apim_cfg.get('endpoint')
    if not endpoint:
        raise ValueError(
            "APIM endpoint not set. Provide via config apim.endpoint or APIM_ENDPOINT env var."
        )

    client = AsyncOpenAI(base_url=endpoint, api_key=api_key)
    done_urls = load_checkpoint(output_path)

    logger.info(f"Loading {input_csv} ...")
    df = pd.read_csv(input_csv, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows")

    df = df.dropna(subset=[URL_COL, TEXT_COL])
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df = df[df[TEXT_COL].str.split().str.len() >= min_words]
    logger.info(f"After ≥{min_words}-word filter: {len(df):,} rows")

    df = df[~df[URL_COL].isin(done_urls)]
    total = len(df)
    logger.info(f"After checkpoint skip: {total:,} remaining to process")

    if total == 0:
        logger.info("Nothing to do.")
        return

    counters: dict = {'done': 0, 'errors': 0}
    model_idx: list = [0]  # mutable round-robin counter, safe in single-threaded asyncio

    # Queue with backpressure so we don't hold all rows in memory
    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 4)

    with open(output_path, 'a') as out_fh:
        worker_tasks = [
            asyncio.create_task(
                worker(queue, client, models, model_idx, max_chars, out_fh, counters)
            )
            for _ in range(concurrency)
        ]
        stats_task = asyncio.create_task(stats_loop(counters, total))

        # Feed queue; await blocks when full, yielding to workers
        for _, row in df.iterrows():
            await queue.put((row[URL_COL], str(row.get(AUTH_COL, '')), row[TEXT_COL]))

        # Poison pills to stop each worker
        for _ in range(concurrency):
            await queue.put(None)

        await asyncio.gather(*worker_tasks)
        stats_task.cancel()

    logger.info(
        f"Complete — {counters['done']:,} processed, {counters['errors']:,} errors | "
        f"output: {output_path}"
    )


def _find_latest_csv(output_dir: str) -> str:
    """Find the most recently modified CSV in the crawl output directory."""
    import glob
    csvs = sorted(
        glob.glob(f"{output_dir}/**/*.csv", recursive=True),
        key=os.path.getmtime,
    )
    if not csvs:
        raise FileNotFoundError(f"No CSV found under {output_dir}")
    latest = csvs[-1]
    logger.info(f"Auto-selected input: {latest}")
    return latest


def main():
    p = argparse.ArgumentParser(
        description='Stage 2: Summarise and classify LA documents via Azure APIM'
    )
    p.add_argument('--config', default='config/pipeline.yaml')
    p.add_argument('--input', default=None,
                   help='Input CSV path (default: latest CSV in crawl output_dir)')
    p.add_argument('--output', default=None,
                   help='Output JSONL path (default: from config summarise.output_file)')
    p.add_argument('--concurrency', type=int, default=20,
                   help='Concurrent API requests (default: 20)')
    args = p.parse_args()

    cfg = load_config(args.config)
    input_csv = args.input or _find_latest_csv(cfg['crawl']['output_dir'])
    output_file = args.output or cfg['summarise']['output_file']

    asyncio.run(run(cfg, input_csv, output_file, args.concurrency))


if __name__ == '__main__':
    main()
