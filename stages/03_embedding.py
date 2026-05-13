"""
Stage 3: Embed document summaries using text-embedding-3-large via Azure APIM.

Reads summarise-stage JSONL, embeds the `summary` field in batches, and writes
a new JSONL with an `embedding` field appended to each record.

Supports checkpoint/resume — URLs already present in the output JSONL are skipped.
"""
import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

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

MODEL = "text-embedding-3-large"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_client() -> AsyncOpenAI:
    from ladi.apim import build_embedding_client
    return build_embedding_client()


def load_checkpoint(output_path: Path) -> set:
    """Return set of URLs already embedded in the output JSONL."""
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
        logger.info(f"Checkpoint: {len(done):,} already embedded — resuming")
    return done


def load_summaries(input_path: Path) -> list[dict]:
    """Load all records from the summarise-stage JSONL."""
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Skip error records from summarise stage
                if 'error' in obj and 'summary' not in obj:
                    continue
                records.append(obj)
            except json.JSONDecodeError:
                pass
    return records


async def embed_batch(client: AsyncOpenAI, batch: list[dict], input_field: str) -> list[dict]:
    """Call embeddings API for a batch, return records with embedding added."""
    texts = [str(r.get(input_field, '')) for r in batch]
    resp = await client.embeddings.create(
        model=MODEL,
        input=texts,
    )
    return [{**record, "embedding": emb.embedding}
            for record, emb in zip(batch, resp.data)]


async def worker(
    queue: asyncio.Queue,
    client: AsyncOpenAI,
    out_fh,
    counters: dict,
    input_field: str,
) -> None:
    while True:
        batch = await queue.get()
        if batch is None:
            queue.task_done()
            break

        try:
            results = await embed_batch(client, batch, input_field)
            for r in results:
                out_fh.write(json.dumps(r) + '\n')
            out_fh.flush()
            counters['done'] += len(results)
        except Exception as e:
            if "429" in str(e):
                counters['rate_limits'] += 1
                logger.warning("Rate limited — waiting 60s...")
                await asyncio.sleep(60)
                try:
                    results = await embed_batch(client, batch, input_field)
                    for r in results:
                        out_fh.write(json.dumps(r) + '\n')
                    out_fh.flush()
                    counters['done'] += len(results)
                except Exception as e2:
                    counters['errors'] += len(batch)
                    logger.warning(f"Batch retry failed ({len(batch)} docs): {e2}")
            else:
                counters['errors'] += len(batch)
                urls = [r.get('url', '')[:60] for r in batch[:3]]
                logger.warning(f"Batch failed ({len(batch)} docs) {type(e).__name__}: {e} | urls: {urls}")

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


async def run(cfg: dict, input_file: str, output_file: str, concurrency: int) -> None:
    embed_cfg = cfg.get('embed', {})
    input_field: str = embed_cfg.get('input_field', 'summary')
    batch_size: int = embed_cfg.get('batch_size', 100)

    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    client = build_client()
    done_urls = load_checkpoint(output_path)

    logger.info(f"Loading {input_path} ...")
    records = load_summaries(input_path)
    logger.info(f"Loaded {len(records):,} records from summarise stage")

    records = [r for r in records if r.get('url') not in done_urls]
    total = len(records)
    logger.info(f"After checkpoint skip: {total:,} remaining to embed")

    if total == 0:
        logger.info("Nothing to do.")
        return

    # Split into batches
    batches = [records[i:i + batch_size] for i in range(0, total, batch_size)]
    logger.info(f"Processing {total:,} docs in {len(batches):,} batches of ≤{batch_size}")

    counters: dict = {'done': 0, 'errors': 0, 'rate_limits': 0}
    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 4)

    with open(output_path, 'a') as out_fh:
        worker_tasks = [
            asyncio.create_task(worker(queue, client, out_fh, counters, input_field))
            for _ in range(concurrency)
        ]
        stats_task = asyncio.create_task(stats_loop(counters, total))

        for batch in batches:
            await queue.put(batch)

        for _ in range(concurrency):
            await queue.put(None)

        await asyncio.gather(*worker_tasks)
        stats_task.cancel()

    logger.info(
        f"Complete — {counters['done']:,} embedded, {counters['errors']:,} errors | "
        f"output: {output_path}"
    )


def main():
    p = argparse.ArgumentParser(
        description='Stage 3: Embed document summaries via Azure APIM text-embedding-3-large'
    )
    p.add_argument('--config', default='config/pipeline.yaml')
    p.add_argument('--input', default=None,
                   help='Input JSONL from summarise stage (default: from config summarise.output_file)')
    p.add_argument('--output', default=None,
                   help='Output JSONL path (default: from config embed.output_file)')
    p.add_argument('--concurrency', type=int, default=5,
                   help='Concurrent batch requests (default: 5)')
    args = p.parse_args()

    cfg = load_config(args.config)
    input_file = args.input or cfg['summarise']['output_file']
    output_file = args.output or cfg['embed']['output_file']

    asyncio.run(run(cfg, input_file, output_file, args.concurrency))


if __name__ == '__main__':
    main()
