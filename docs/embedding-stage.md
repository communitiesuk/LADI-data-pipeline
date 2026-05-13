# Embedding Stage — Overview

## What it does

Takes structured summaries from the summarise stage and generates dense vector embeddings for semantic search. Each document's summary is embedded using `text-embedding-3-large` (3072 dimensions), enabling similarity search, clustering, and retrieval.

---

## How it works

### Architecture

```
Summarise JSONL (url, authority, title, year, summary, themes[])
        │
        ▼
┌─────────────────────────────────┐
│   03_embedding.py               │
│                                 │
│  • Load JSONL, skip errors      │
│  • Checkpoint-resume (by URL)   │
│  • Batch requests (100/batch)   │
│  • Async queue + N workers      │
│  • 429 backoff + retry          │
└─────────────────────────────────┘
        │
        ▼
Output JSONL (all fields + embedding[3072])
```

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| Batch API calls (100 docs/batch) | Embeddings API designed for batch input — ~100x fewer HTTP calls |
| Embed the summary, not full text | Summary is concise (2-3 sentences), captures intent; full text adds noise and 100x token cost |
| Checkpoint-resume via JSONL | Can crash/stop and resume without re-embedding |
| Async queue with 3-5 workers | Parallelise batch requests against rate limits |
| Skip error records from summarise | Records with `error` field and no `summary` are excluded |

### Model

- **text-embedding-3-large** via Azure APIM
- 3072 dimensions (full resolution)
- Unit-normed vectors (cosine similarity = dot product)

---

## Evaluation

We validated embedding quality on a **499-document sample** spanning 52 local authorities and all 16 themes.

### Method

| Metric | What it measures | Pass threshold |
|--------|-----------------|----------------|
| **Sanity checks** | Correct dimensions, no zero/NaN vectors | All 3072-dim, no degenerate vectors |
| **Theme coherence** | Within-theme similarity vs between-theme | Ratio > 1.0 |
| **k-NN theme precision** | % of top-k neighbors sharing ≥1 theme | Mean > 0.50 |
| **Authority coherence** | Within-authority similarity vs between-authority | Ratio > 1.0 |

### Results (n=499)

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Dimensions | 3072 | 3072 | PASS |
| Zero/NaN vectors | 0 | 0 | PASS |
| Within-theme avg similarity | **0.372** | — | — |
| Between-theme avg similarity | 0.304 | — | — |
| Theme coherence ratio | **1.22x** | > 1.0 | PASS |
| k-NN theme precision (k=5) | **76%** | > 50% | PASS |
| Authority coherence ratio | **1.33x** | > 1.0 | PASS |

### Per-theme breakdown

| Theme | n docs | Avg within-theme similarity |
|-------|--------|----------------------------|
| Children and young families | 8 | 0.431 |
| Local government finance | 67 | 0.414 |
| Housing | 25 | 0.390 |
| Advice and benefits | 10 | 0.381 |
| Adult social care | 5 | 0.375 |
| Council and democracy | 239 | 0.372 |
| Community safety | 30 | 0.371 |
| Transport and highways | 27 | 0.367 |
| People and communities | 22 | 0.366 |
| Planning and development | 142 | 0.364 |
| Environment and waste | 38 | 0.360 |
| Education | 31 | 0.348 |
| Leisure and culture | 11 | 0.330 |
| Business and employment | 25 | 0.326 |
| Public health | 12 | 0.289 |

All themes show higher within-theme similarity than the between-theme baseline (0.304), with specialist topics (children, finance, housing) clustering most tightly.

---

## Search quality test

We tested retrieval quality with natural-language queries against the 499-document index.

### Example searches

**Query: "housing strategy affordable homes"**

| Rank | Title | Authority | Sim |
|------|-------|-----------|-----|
| 1 | Blaby District Council Housing Strategy 2020-2025 | blaby.gov.uk | 0.583 |
| 2 | Housing Position Statement – Cabinet Report | Devon County Council | 0.492 |
| 3 | Housing Advice for Single People in Havant | Havant Borough Council | 0.449 |
| 4 | Westminster City Council Tenancy Strategy 2025 | westminster.gov.uk | 0.440 |
| 5 | Statement on Housing Delivery (Oadby and Wigston) | Oadby and Wigston BC | 0.437 |

All 5 results are housing-themed. Top hit is a housing strategy document.

**Query: "council budget cuts financial savings"**

| Rank | Title | Authority | Sim |
|------|-------|-----------|-----|
| 1 | Appendix A: Savings Proposals for Approval | LB Redbridge | 0.603 |
| 2 | Saving Proposals Identified - 2026/27 Budget Cycle | Exeter City Council | 0.584 |
| 3 | Revised Budget – Final Settlement for 2016-17 | Suffolk County Council | 0.537 |
| 4 | The 2024/25 Budget and MTFS | Southampton City Council | 0.534 |
| 5 | Redbridge Council Proposed Budget 2024/25 | LB Redbridge | 0.508 |

All 5 results are finance-themed with high similarity scores (>0.5).

**Query: "climate change net zero carbon emissions"**

| Rank | Title | Authority | Sim |
|------|-------|-----------|-----|
| 1 | Representation on Exeter Plan Chapter 5: Climate Change (CC1: Net Zero) | Exeter City Council | 0.517 |
| 2 | Cabinet Lead for Climate Emergency Update | Havant Borough Council | 0.388 |
| 3 | Climate Change Cabinet Advisory Group Agenda | Thanet District Council | 0.344 |
| 4 | Council Position Statement on 3G Artificial Pitches | Rochford District Council | 0.269 |
| 5 | Suffolk CC Motion on Increasing Tree Cover | Suffolk County Council | 0.261 |

Top 3 are direct climate policy documents. Clear similarity drop-off after the most relevant results.

**Query: "school places primary education children"**

| Rank | Title | Authority | Sim |
|------|-------|-----------|-----|
| 1 | Cambridgeshire Junior Year Allocation Information | Cambridgeshire CC | 0.466 |
| 2 | Approval for Public Notice: Crown Meadow School Age Range | worcestershire.gov.uk | 0.399 |
| 3 | Welcome to Secondary School: A Guide for Pupils | LB Hillingdon | 0.385 |
| 4 | High Littleton Primary School Admission Appeal Form | bathnes.gov.uk | 0.375 |
| 5 | Local Authority Report to The Schools Adjudicator 2023 | LB Newham | 0.375 |

All 5 results are education-themed.

### Recall test: Infrastructure Funding Statements

The dataset contains exactly **5 IFS documents**. Searching for "infrastructure funding statement" returns all 5 as the top 5 results — **100% recall**.

| Rank | Title | Authority | Sim |
|------|-------|-----------|-----|
| 1 | Gloucester City Council Annual IFS 2024-25 | gloucester.moderngov.co.uk | 0.576 |
| 2 | Infrastructure Funding Statement – Priorities Report | sevenoaks.moderngov.co.uk | 0.527 |
| 3 | Gloucestershire CC Annual IFS 2023/24 | glostext.gloucestershire.gov.uk | 0.519 |
| 4 | South Ribble BC Annual IFS 2024/25 | southribble.moderngov.co.uk | 0.446 |
| 5 | City of York Council IFS 2019-20 | york.gov.uk | 0.444 |

---

## Performance

| Metric | Value |
|--------|-------|
| 499 documents | 9 seconds (5 batches of 100) |
| Throughput | ~3,300 docs/min |
| Rate limits hit | 0 |
| Errors | 0 |
| Estimated full corpus (773K docs) | ~4 hours |

---

## Files

| File | Purpose |
|------|---------|
| `stages/03_embedding.py` | Main pipeline stage — batched embedding with checkpoint-resume |
| `eval_embedding.py` | Evaluation script — theme coherence, k-NN precision, authority coherence |
| `search_demo.py` | Interactive search demo — embed a query and find top-k results |
| `data/embeddings_sample.jsonl` | 499 embedded documents (sample) |
| `config/pipeline.yaml` | Pipeline configuration (embed section) |

---

## Running

```bash
# Embed the full summarise output
python stages/03_embedding.py --concurrency 5

# Embed a specific file
python stages/03_embedding.py --input data/summaries_sample.jsonl --output data/embeddings_sample.jsonl

# Run evaluation
python eval_embedding.py --input data/embeddings_sample.jsonl

# Search interactively
python search_demo.py --interactive

# Single query
python search_demo.py "infrastructure funding statement"
```

Requires environment variables:
- `LADI_APIM_EMBEDDING_URL`
- `LADI_APIM_SUBSCRIPTION_KEY`
- `LADI_APIM_API_VERSION`
- `LADI_APIM_TOKEN_SCOPE`
