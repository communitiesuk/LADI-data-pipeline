# Summarise Stage — Overview

## What it does

Takes raw extracted text from ~1.27M crawled UK local authority documents and produces structured metadata for each one:

- **Title** — a concise document title
- **Year** — publication or financial year
- **Summary** — 2-3 sentence description suitable for search
- **Themes** — 1-2 topic classifications from a fixed 16-theme taxonomy

This structured output powers downstream search, filtering, and embedding stages.

---

## How it works

### Architecture

```
Crawl CSV (url, authority, text)
        │
        ▼
┌─────────────────────────────┐
│   02_summarise.py           │
│                             │
│  • Filter: ≥100 words       │
│  • Clean: strip ctrl chars  │
│  • Truncate: 600 words max  │
│  • GPT-4.1 structured call  │
│  • Checkpoint-resume        │
│  • Async queue + N workers  │
└─────────────────────────────┘
        │
        ▼
Output JSONL (url, authority, title, year, summary, themes[])
```

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| Async queue with 10-20 workers | Maximise throughput against Azure APIM rate limits |
| Checkpoint-resume via JSONL | Process can crash/stop and resume without re-doing work |
| 600-word truncation | Keeps token usage low while retaining enough context for classification |
| JSON response format | Structured output, no parsing fragility |
| Single model call per doc | Title + year + summary + themes in one shot (cheaper than separate calls) |
| 1-2 themes max | Prevents over-classification; "only 2 if the doc genuinely fits both" |

### Model

- **GPT-4.1** via Azure APIM (Azure OpenAI)
- `temperature=0.0`, `max_tokens=300`
- `response_format: json_object`

### Theme taxonomy (16 categories)

| | | |
|---|---|---|
| Advice and benefits | Adult social care | Business and employment |
| Community safety | Council and democracy | Children and young families |
| Education | Environment and waste | Housing |
| Leisure and culture | Local government finance | People and communities |
| Planning and development | Public health | Transport and highways |
| Uncategorised | | |

---

## Evaluation

We validated the summarise stage against a manually-annotated gold standard of **59 documents** spanning all themes, including 5 Infrastructure Funding Statement (IFS) documents (the priority use case).

### Method

Each document is scored on 4 dimensions:

| Dimension | Method | What "good" means |
|-----------|--------|-------------------|
| **Themes** | Automatic comparison | Gold primary theme appears in predicted themes |
| **Year** | Automatic comparison | Predicted year within ±1 of gold (handles 2024/25 ambiguity) |
| **Title** | LLM judge (1-5 scale) | Accurately identifies the same document |
| **Summary** | LLM judge (1-5 scale) | Faithful, specific, useful for search |

### Results

#### Overall (n=59)

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Theme accuracy (primary match) | **88%** | ≥75% | PASS |
| Year accuracy (off-by-one) | **95%** | ≥75% | PASS |
| Title quality (mean) | **4.5/5** | ≥3.5 | PASS |
| Summary quality (mean) | **4.6/5** | ≥3.5 | PASS |

- Summary fail rate (score <3): **0%**
- Title fail rate (score <3): **2%** (1 document — an appendix where the model titled the sub-section rather than the parent document)

#### IFS documents (n=5) — higher bar

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Theme accuracy | **100%** | ≥85% | PASS |
| Year accuracy | **100%** | ≥80% | PASS |
| Title quality | **4.6/5** | ≥4.0 | PASS |
| Summary quality | **4.2/5** | ≥4.0 | PASS |

#### Per-theme breakdown

| Theme | Accuracy |
|-------|----------|
| Planning and development | 19/19 (100%) |
| Education | 7/7 (100%) |
| Environment and waste | 2/2 (100%) |
| Housing | 3/3 (100%) |
| Community safety | 1/1 (100%) |
| Local government finance | 1/1 (100%) |
| People and communities | 1/1 (100%) |
| Council and democracy | 13/15 (87%) |
| Business and employment | 3/4 (75%) |
| Transport and highways | 1/2 (50%) |
| Public health | 1/3 (33%) |
| Advice and benefits | 0/1 (0%) |

Themes with lower accuracy have very small sample sizes (1-3 docs). The two highest-volume themes (Planning and development, Council and democracy) perform well.

---

## Scale estimates

| Metric | Value |
|--------|-------|
| Total documents | ~773K (after filtering ≥100 words from 1.27M crawled) |
| Input tokens per doc | ~900 (600 words + system prompt) |
| Output tokens per doc | ~200 |
| Rate limit | 8K TPM per model (Azure APIM) |
| Throughput | ~90 docs/min with round-robin across 3 models |
| Estimated runtime | ~4 days for full corpus |

---

## Files

| File | Purpose |
|------|---------|
| `stages/02_summarise.py` | Main pipeline stage — processes CSVs, writes JSONL |
| `eval_summarise.py` | Evaluation script — runs gold standard test, reports pass/fail |
| `eval_gold_standard.csv` | 59 manually-annotated test documents |
| `eval_summarise_results.csv` | Full per-document evaluation results |
| `config/pipeline.yaml` | Pipeline configuration |

---

## Running

```bash
# Run summarise on crawl output
python stages/02_summarise.py --concurrency 20

# Run evaluation
python eval_summarise.py
```

Requires environment variables:
- `LADI_APIM_BASE_URL`
- `LADI_APIM_SUBSCRIPTION_KEY`
- `LADI_APIM_API_VERSION`
- `LADI_APIM_TOKEN_SCOPE`
