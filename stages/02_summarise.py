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

DOC_TYPES = [
    # Planning
    "Local Plan",
    "Neighbourhood Plan",
    "Community Infrastructure Levy",
    "Infrastructure Funding Statement",
    "Sustainability Appraisal",
    "Consultation Statement",
    "Flood Risk Assessment",
    "Housing Needs Assessment",
    "Transport Plan",
    # Financial (all statutory)
    "Statement of Accounts",
    "Medium Term Financial Strategy",
    "Treasury Management Strategy",
    "Capital Strategy",
    "Annual Governance Statement",
    "Fees and Charges Schedule",
    "Internal Audit Report",
    # Policy / Strategy
    "Climate Action Plan",
    "Equality Impact Assessment",
    "Asset Management Plan",
    "Corporate Plan",
    "Licensing Policy",
    "Health and Wellbeing Strategy",
    "Anti-Social Behaviour Strategy",
    "Code of Conduct",
    "Public Health Annual Report",
    # Catch-all
    "other",
]

SYSTEM_PROMPT = f"""You extract metadata from UK local authority documents.

Given document text, return JSON with exactly these fields:
- title: string (document title; infer from content if not stated)
- year: integer or null (publication year; null if unknown)
- summary: string (2-3 sentences describing purpose and key content, suitable for search)
- themes: array of 1-2 themes from the list below
- doc_type: string — the single best matching document type from the list below; use "other" if none fit OR if this document merely discusses, references, or responds to one of those document types rather than being that document itself (e.g. an inspector's letter about a Local Plan → "other"; a consultation response about a CIL → "other")

Only give 2 themes if the document genuinely fits both. E.g, a referendum on a neighbourhood plan is both "Council and democracy" and "Planning and development".

Themes:
{chr(10).join(f'- {t}' for t in THEMES)}

Document types:
{chr(10).join(f'- {d}' for d in DOC_TYPES)}

Respond with valid JSON only. No markdown fences."""

# CSV column names from crawldocs output
URL_COL = 'Document Link'
TEXT_COL = 'text'
AUTH_COL = 'Authority Name'

# Model profiles — controls which env vars and API params are used
MODEL_PROFILES = {
    # Sandbox / legacy
    'gpt4.1': {
        'deployment_id': 'gpt-4-1',
        'is_reasoning': False,
        'base_url_env': 'LADI_APIM_BASE_URL',
        'key_env': 'LADI_APIM_SUBSCRIPTION_KEY',
        'max_output_tokens': 300,
    },
    'gpt5mini-sandbox': {
        'deployment_id': 'gpt-5-mini',
        'is_reasoning': True,
        'base_url_env': 'LADI_APIM_BASE_URL_5MINI',
        'key_env': 'LADI_APIM_SUBSCRIPTION_KEY_5MINI',
        'max_output_tokens': 1000,
    },
    # Production LADI endpoints (AD token from MHCLG tenant + subscription key)
    'gpt5.1': {
        'deployment_id': 'gpt5-1',
        'is_reasoning': True,
        'base_url_env': 'LADI_PROD_APIM_BASE_URL_51',
        'key_env': 'LADI_PROD_APIM_KEY',
        'api_version_env': 'LADI_PROD_APIM_API_VERSION',
        'token_scope_env': 'LADI_PROD_APIM_TOKEN_SCOPE',
        'max_output_tokens': 1000,
        'use_ad_token': True,
    },
    'gpt5mini': {
        'deployment_id': 'gpt-5-mini',
        'is_reasoning': True,
        'base_url_env': 'LADI_PROD_APIM_BASE_URL_5MINI',
        'key_env': 'LADI_PROD_APIM_KEY',
        'api_version_env': 'LADI_PROD_APIM_API_VERSION',
        'token_scope_env': 'LADI_PROD_APIM_TOKEN_SCOPE',
        'max_output_tokens': 1000,
        'use_ad_token': True,
    },
    'gpt5nano': {
        'deployment_id': 'gpt-5-nano',
        'is_reasoning': True,
        'base_url_env': 'LADI_PROD_APIM_BASE_URL_5NANO',
        'key_env': 'LADI_PROD_APIM_KEY',
        'api_version_env': 'LADI_PROD_APIM_API_VERSION',
        'token_scope_env': 'LADI_PROD_APIM_TOKEN_SCOPE',
        'max_output_tokens': 2000,  # nano burns heavily on reasoning tokens
        'use_ad_token': True,
    },
}


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


def build_client(profile: dict) -> AsyncOpenAI:
    import subprocess, json as _json
    sub_key = os.environ[profile['key_env']]
    if profile.get('use_ad_token', True):
        scope_env = profile.get('token_scope_env', 'LADI_APIM_TOKEN_SCOPE')
        scope = os.environ[scope_env]
        result = subprocess.run(
            ["az", "account", "get-access-token", "--scope", scope, "--output", "json"],
            check=True, capture_output=True, text=True,
        )
        api_key = _json.loads(result.stdout)["accessToken"]
    else:
        api_key = sub_key
    api_version_env = profile.get('api_version_env', 'LADI_APIM_API_VERSION')
    return AsyncOpenAI(
        base_url=os.environ[profile['base_url_env']],
        api_key=api_key,
        default_headers={"Ocp-Apim-Subscription-Key": sub_key},
        default_query={"api-version": os.environ[api_version_env]},
        max_retries=0,
        timeout=60.0,
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


async def call_api(client: AsyncOpenAI, url: str, authority: str, text: str, profile: dict) -> dict:
    cleaned = clean_text(truncate_words(text))
    # Azure WAF sanitisation — colons trigger PII/geolocation rules, strip other risky content
    cleaned = cleaned.replace(":", " -")
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"www\.\S+", "", cleaned)
    cleaned = re.sub(r"[\w.-]+@[\w.-]+\.\w+", "", cleaned)
    cleaned = re.sub(r"\d{4,5}\s?\d{6}", "", cleaned)  # UK phone numbers
    cleaned = re.sub(r"[^\x00-\x7F]+", "", cleaned)  # non-ASCII (Cyrillic etc.)
    cleaned = re.sub(r"\b(database|schema)\s*\(", r"\1 ", cleaned)  # SQL injection pattern
    cleaned = re.sub(r"(\d)\s+as\s+(\w+)\s+from", r"\1 as \2 since", cleaned)  # SQL SELECT pattern
    # Skip garbled text (bad PDF extraction) — if <50% alpha chars, not useful
    alpha_ratio = sum(c.isalpha() for c in cleaned) / max(len(cleaned), 1)
    if alpha_ratio < 0.5:
        return {"url": url, "authority": authority, "error": "garbled_text",
                "_input_tokens": 0, "_output_tokens": 0, "_reasoning_tokens": 0}
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
    result = json.loads(resp.choices[0].message.content)
    result.update({'url': url, 'authority': authority})

    # Token accounting
    usage = resp.usage
    result['_input_tokens'] = usage.prompt_tokens if usage else 0
    result['_output_tokens'] = usage.completion_tokens if usage else 0
    details = getattr(usage, 'completion_tokens_details', None) if usage else None
    result['_reasoning_tokens'] = getattr(details, 'reasoning_tokens', 0) or 0

    return result


async def worker(
    queue: asyncio.Queue,
    client_ref: dict,
    out_fh,
    counters: dict,
    profile: dict,
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        url, authority, text = item
        backoff = 60

        while True:
            # Honour shared rate-limit pause — set by whichever worker last hit 429
            wait = counters['backoff_until'] - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)

            try:
                result = await call_api(client_ref['client'], url, authority, text, profile)

                # Move internal token fields to per-doc output fields and accumulate
                in_tok = result.pop('_input_tokens', 0)
                out_tok = result.pop('_output_tokens', 0)
                reason_tok = result.pop('_reasoning_tokens', 0)
                counters['input_tokens'] += in_tok
                counters['output_tokens'] += out_tok
                counters['reasoning_tokens'] += reason_tok
                result['input_tokens'] = in_tok
                result['output_tokens'] = out_tok
                if reason_tok:
                    result['reasoning_tokens'] = reason_tok

                out_fh.write(json.dumps(result) + '\n')
                out_fh.flush()
                counters['done'] += 1
                backoff = 60  # reset per-worker backoff on success
                break

            except Exception as e:
                if "429" in str(e):
                    counters['rate_limits'] += 1
                    # Shared backoff: push the global pause-until forward
                    new_until = time.monotonic() + backoff
                    if new_until > counters['backoff_until']:
                        counters['backoff_until'] = new_until
                        logger.warning(f"Rate limited — shared backoff {backoff}s (will retry {url[:60]})")
                    backoff = min(backoff * 2, 300)
                elif "401" in str(e) and profile.get('use_ad_token'):
                    logger.warning("401 Unauthorized — refreshing AD token and retrying ...")
                    try:
                        client_ref['client'] = build_client(profile)
                        logger.info("Token refreshed after 401.")
                    except Exception as ref_e:
                        logger.warning(f"Token refresh failed: {ref_e}")
                    await asyncio.sleep(5)
                    # falls through to retry the inner while loop
                else:
                    counters['errors'] += 1
                    logger.warning(f"[{url[:80]}] {type(e).__name__}: {e}")
                    break  # real error — skip this doc

        queue.task_done()


async def token_refresh_loop(client_ref: dict, profile: dict, interval: int = 2700) -> None:
    """Refresh the Azure AD token every 45 minutes to prevent mid-run 401s."""
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                logger.info("Refreshing Azure AD token ...")
                client_ref['client'] = build_client(profile)
                logger.info("Azure AD token refreshed successfully.")
            except Exception as e:
                logger.warning(f"Token refresh failed: {e} — will retry next cycle")
    except asyncio.CancelledError:
        pass


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

            in_tok = counters['input_tokens']
            out_tok = counters['output_tokens']
            reason_tok = counters['reasoning_tokens']
            avg_in = in_tok // done if done else 0
            avg_out = out_tok // done if done else 0

            logger.info(
                f"Progress: {done:,}/{total:,} done | {errors:,} errors | "
                f"{counters['rate_limits']} rate limits | "
                f"{rate:.0f} docs/min | ETA {eta}"
            )
            logger.info(
                f"Tokens:   in {in_tok:,} total ({avg_in}/doc) | "
                f"out {out_tok:,} total ({avg_out}/doc)"
                + (f" | reasoning {reason_tok:,} total" if reason_tok else "")
            )
    except asyncio.CancelledError:
        pass


async def run(cfg: dict, input_csv: str, output_file: str, concurrency: int, profile: dict, limit: int = None) -> None:
    sum_cfg = cfg['summarise']
    min_words: int = sum_cfg.get('min_words', 100)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client_ref = {'client': build_client(profile)}
    done_urls = load_checkpoint(output_path)

    logger.info(f"Loading {input_csv} ...")
    df = pd.read_csv(input_csv, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows")

    df = df.dropna(subset=[URL_COL, TEXT_COL])
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df = df[df[TEXT_COL].str.split().str.len() >= min_words]
    logger.info(f"After ≥{min_words}-word filter: {len(df):,} rows")

    df = df[~df[URL_COL].isin(done_urls)]
    if limit:
        df = df.head(limit)
        logger.info(f"Limiting to first {limit:,} docs")
    total = len(df)
    logger.info(f"After checkpoint skip: {total:,} remaining to process")

    if total == 0:
        logger.info("Nothing to do.")
        return

    counters: dict = {
        'done': 0, 'errors': 0, 'rate_limits': 0,
        'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0,
        'backoff_until': 0.0,  # shared rate-limit pause timestamp
    }

    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 4)

    with open(output_path, 'a') as out_fh:
        worker_tasks = [
            asyncio.create_task(worker(queue, client_ref, out_fh, counters, profile))
            for _ in range(concurrency)
        ]
        stats_task = asyncio.create_task(stats_loop(counters, total))
        refresh_task = (
            asyncio.create_task(token_refresh_loop(client_ref, profile))
            if profile.get('use_ad_token') else None
        )

        for _, row in df.iterrows():
            await queue.put((row[URL_COL], str(row.get(AUTH_COL, '')), row[TEXT_COL]))

        for _ in range(concurrency):
            await queue.put(None)

        await asyncio.gather(*worker_tasks)
        stats_task.cancel()
        if refresh_task:
            refresh_task.cancel()

    done = counters['done']
    in_tok = counters['input_tokens']
    out_tok = counters['output_tokens']
    reason_tok = counters['reasoning_tokens']
    logger.info(
        f"Complete — {done:,} processed, {counters['errors']:,} errors, "
        f"{counters['rate_limits']} rate limits | output: {output_path}"
    )
    logger.info(
        f"Tokens — input: {in_tok:,} ({in_tok // done if done else 0}/doc) | "
        f"output: {out_tok:,} ({out_tok // done if done else 0}/doc)"
        + (f" | reasoning: {reason_tok:,} ({reason_tok // done if done else 0}/doc)" if reason_tok else "")
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
    p.add_argument('--model', default='gpt4.1', choices=list(MODEL_PROFILES),
                   help='Model profile to use (default: gpt4.1)')
    p.add_argument('--limit', type=int, default=None,
                   help='Process only the first N docs (for testing)')
    args = p.parse_args()

    profile = MODEL_PROFILES[args.model]
    logger.info(f"Using model profile: {args.model} → deployment={profile['deployment_id']}, reasoning={profile['is_reasoning']}")

    cfg = load_config(args.config)
    input_csv = args.input or _find_latest_csv(cfg['crawl']['output_dir'])
    output_file = args.output or cfg['summarise']['output_file']

    asyncio.run(run(cfg, input_csv, output_file, args.concurrency, profile, limit=args.limit))


if __name__ == '__main__':
    main()
