"""
Stage 2: Summarise and classify documents using GPT via Azure APIM.

For each document with ≥100 words, calls GPT to extract:
  title, year, summary, themes (1-2)

Output: JSONL (one JSON object per line, appended incrementally).
Supports checkpoint/resume — already-processed URLs are skipped on restart.
"""
import argparse
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

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

SYSTEM_PROMPT = f"""You extract metadata from UK local authority documents.

Given document text, return JSON with exactly these fields:
- title: string (document title; infer from content if not stated)
- year: integer or null (publication year; null if unknown)
- summary: string (2-3 sentences describing purpose and key content, suitable for search)
- themes: array of 1-2 themes from the list below

Only give 2 themes if the document genuinely fits both — not just because it mentions a related topic. For example, a referendum on a neighbourhood plan is both "Council and democracy" and "Planning and development" because it is genuinely a democratic process about a planning matter.

Themes:
{chr(10).join(f'- {t}' for t in THEMES)}

Respond with valid JSON only. No markdown fences."""

# CSV column names from crawldocs output
URL_COL = 'Document Link'
TEXT_COL = 'text'
AUTH_COL = 'Authority Name'


def clean_text(text: str) -> str:
    """Strip invisible control characters that cause APIM 403s."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


def truncate_words(text: str, max_words: int = 600) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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
                pass
    if done:
        logger.info(f"Checkpoint: {len(done):,} already processed — resuming")
    return done


async def call_api(client: AsyncOpenAI, url: str, authority: str, text: str) -> dict:
    cleaned = clean_text(truncate_words(text))
    user_content = f"URL: {url}\n\n{cleaned}"

    resp = await client.chat.completions.create(
        model="gpt-4-1",
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_content},
        ],
        response_format={'type': 'json_object'},
        temperature=0.0,
        max_tokens=300,
    )
    result = json.loads(resp.choices[0].message.content)
    result.update({'url': url, 'authority': authority})
    return result


async def worker(
    queue: asyncio.Queue,
    client: AsyncOpenAI,
    out_fh,
    counters: dict,
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        url, authority, text = item

        try:
            result = await call_api(client, url, authority, text)
            out_fh.write(json.dumps(result) + '\n')
            out_fh.flush()
            counters['done'] += 1
        except Exception as e:
            if "429" in str(e):
                counters['rate_limits'] += 1
                logger.warning(f"Rate limited — waiting 60s...")
                await asyncio.sleep(60)
                try:
                    result = await call_api(client, url, authority, text)
                    out_fh.write(json.dumps(result) + '\n')
                    out_fh.flush()
                    counters['done'] += 1
                except Exception as e2:
                    counters['errors'] += 1
                    logger.warning(f"[{url[:80]}] retry failed: {e2}")
            else:
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
                f"{counters['rate_limits']} rate limits | "
                f"{rate:.0f} docs/min | ETA {eta}"
            )
    except asyncio.CancelledError:
        pass


async def run(cfg: dict, input_csv: str, output_file: str, concurrency: int) -> None:
    sum_cfg = cfg['summarise']
    min_words: int = sum_cfg.get('min_words', 100)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = build_client()
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

    counters: dict = {'done': 0, 'errors': 0, 'rate_limits': 0}

    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 4)

    with open(output_path, 'a') as out_fh:
        worker_tasks = [
            asyncio.create_task(worker(queue, client, out_fh, counters))
            for _ in range(concurrency)
        ]
        stats_task = asyncio.create_task(stats_loop(counters, total))

        for _, row in df.iterrows():
            await queue.put((row[URL_COL], str(row.get(AUTH_COL, '')), row[TEXT_COL]))

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
    p.add_argument('--concurrency', type=int, default=10,
                   help='Concurrent API requests (default: 10)')
    args = p.parse_args()

    cfg = load_config(args.config)
    input_csv = args.input or _find_latest_csv(cfg['crawl']['output_dir'])
    output_file = args.output or cfg['summarise']['output_file']

    asyncio.run(run(cfg, input_csv, output_file, args.concurrency))


if __name__ == '__main__':
    main()
