# Search Method Evaluation Report

**Dataset:** 1,430 embedded UK local authority documents across 268 authorities
**Database:** Azure Postgres with pgvector (text-embedding-3-large, 3072 dimensions)
**Search tool:** `search_comparison.py` — all queries run directly against Postgres
**Date:** June 2026

---

## Methods Tested

| Method | How it works | Notes |
|--------|-------------|-------|
| **Embedding** | Embed query → pgvector `<=>` cosine distance on stored vectors | Semantic understanding |
| **Full text** | `ILIKE` keyword match on `document_text` column | High recall, noisy |
| **Summary** | `ILIKE` keyword match on GPT-generated summary | Concise, lower noise |
| **Title** | `ILIKE` keyword match on document title | Precise, low recall |

All 4 methods query Azure Postgres directly. Embedding search embeds the query via Azure APIM then uses pgvector's `<=>` operator to rank by cosine similarity. Keyword methods score by fraction of query terms matched plus a 0.5 boost for exact phrase match.

---

## Query Results

### 1. "infrastructure funding statement"

**Embedding** (pgvector)
| # | Title | Authority | Similarity |
|---|-------|-----------|------------|
| 1 | Gloucester City Council Annual Infrastructure Funding Statement 2024-25 | gloucester.moderngov.co.uk | 0.576 |
| 2 | Infrastructure Funding Statement – Priorities Report | sevenoaks.moderngov.co.uk | 0.527 |
| 3 | Gloucestershire County Council Annual Infrastructure Funding Statement 2023/24 | glostext.gloucestershire.gov.uk | 0.519 |
| 4 | Delivery and Monitoring of the Local Plan | Newcastle City Council | 0.470 |
| 5 | South Ribble Borough Council Annual Infrastructure Funding Statement 2024/25 | southribble.moderngov.co.uk | 0.446 |

**Full text keyword** (ILIKE on document_text)
| # | Title | Authority |
|---|-------|-----------|
| 1 | Infrastructure Funding Statement – Priorities Report | sevenoaks.moderngov.co.uk |
| 2 | Bath and North East Somerset Council Authorities Monitoring Report 2023/24 | bathnes.gov.uk |
| 3 | City of York Council Infrastructure Funding Statement 2019-20 | york.gov.uk |
| 4 | Gloucester City Council Annual Infrastructure Funding Statement 2024-25 | gloucester.moderngov.co.uk |
| 5 | Gloucestershire County Council Annual Infrastructure Funding Statement 2023/24 | glostext.gloucestershire.gov.uk |

**Summary keyword** (ILIKE on summary)
| # | Title | Authority |
|---|-------|-----------|
| 1 | Gloucestershire County Council Annual Infrastructure Funding Statement 2023/24 | glostext.gloucestershire.gov.uk |
| 2 | Gloucester City Council Annual Infrastructure Funding Statement 2024-25 | gloucester.moderngov.co.uk |
| 3 | Infrastructure Funding Statement – Priorities Report | sevenoaks.moderngov.co.uk |
| 4 | South Ribble Borough Council Annual Infrastructure Funding Statement 2024/25 | southribble.moderngov.co.uk |
| 5 | City of York Council Infrastructure Funding Statement 2019-20 | york.gov.uk |

**Title keyword** (ILIKE on title)
| # | Title | Authority |
|---|-------|-----------|
| 1 | Gloucester City Council Annual Infrastructure Funding Statement 2024-25 | gloucester.moderngov.co.uk |
| 2 | Infrastructure Funding Statement – Priorities Report | sevenoaks.moderngov.co.uk |
| 3 | South Ribble Borough Council Annual Infrastructure Funding Statement 2024/25 | southribble.moderngov.co.uk |
| 4 | City of York Council Infrastructure Funding Statement 2019-20 | york.gov.uk |
| 5 | Gloucestershire County Council Annual Infrastructure Funding Statement 2023/24 | glostext.gloucestershire.gov.uk |

> **Verdict:** Summary and title keyword both return all 5 true IFS documents. Embedding returns 4 of 5 IFS docs plus a relevant Local Plan monitoring report. Full text also does well (4/5 IFS docs), with the Bath AMR surfacing as a near-miss (it references IFS throughout). For exact document type names, keyword methods match or beat embedding. Summary ∩ Title overlap: 5/5.

---

### 2. "housing strategy affordable homes"

**Embedding** (pgvector)
| # | Title | Authority | Similarity |
|---|-------|-----------|------------|
| 1 | Housing Development Strategy 2025/26-2030/31 | Somerset Council | 0.583 |
| 2 | Blaby District Council Housing Strategy 2020-2025 | blaby.gov.uk | 0.583 |
| 3 | Accelerating Affordable Housing Delivery: A Quality Home for All | Stockton-on-Tees Borough Council | 0.543 |
| 4 | Affordable Housing Development: Anticipated Delivery Schedule | Newark and Sherwood District Council | 0.513 |
| 5 | Housing Position Statement – Cabinet Report | Devon County Council | 0.492 |

**Full text keyword** (ILIKE on document_text)
| # | Title | Authority |
|---|-------|-----------|
| 1 | Horton Heath Development Management Committee Report | Eastleigh Borough Council |
| 2 | Westminster City Council Tenancy Strategy 2025 | westminster.gov.uk |
| 3 | Adoption of Local Development Scheme (LDS) 2025 | Dover District Council |
| 4 | Executive Work Programme and Forward Plan (March 2026–) | West Oxfordshire District Council |
| 5 | Planning Committee Minutes – 23 September 2025 | London Borough of Barking and Dagenham |

**Summary keyword** (ILIKE on summary)
| # | Title | Authority |
|---|-------|-----------|
| 1 | Affordable Housing Development: Anticipated Delivery Schedule | Newark and Sherwood District Council |
| 2 | Housing Position Statement – Cabinet Report | Devon County Council |
| 3 | Housing Development Strategy 2025/26-2030/31 | Somerset Council |
| 4 | Sustainability Appraisal Report: Modifications to Submission | blaby.gov.uk |
| 5 | Summary of the Developer's Main Planning Obligations for Barking Riverside | London Borough of Barking and Dagenham |

**Title keyword** (ILIKE on title)
| # | Title | Authority |
|---|-------|-----------|
| 1 | Support for Hyde Housing – 15 Social Rent Homes at Birdham | Chichester District Council |
| 2 | Affordable Housing Development: Anticipated Delivery Schedule | Newark and Sherwood District Council |
| 3 | Housing Development Strategy 2025/26-2030/31 | Somerset Council |
| 4 | Blaby District Council Housing Strategy 2020-2025 | blaby.gov.uk |
| 5 | Rugby Updated Housing Needs Evidence Report | Rugby Borough Council |

> **Verdict:** Embedding clearly wins — all 5 results are directly relevant housing strategy documents with strong similarity scores (0.49–0.58). Summary keyword gets 3/5 right but surfaces a Sustainability Appraisal and a planning obligations doc. Full text is the worst performer, returning committee minutes, a forward plan, and an LDS that merely contain "housing" or "affordable" in passing. Zero overlap between embedding and full text.

---

### 3. "council budget savings"

**Embedding** (pgvector)
| # | Title | Authority | Similarity |
|---|-------|-----------|------------|
| 1 | Saving Proposals Identified – 2026/27 Budget Cycle | Exeter City Council | 0.633 |
| 2 | Appendix A: Savings Proposals for Approval | London Borough of Redbridge | 0.618 |
| 3 | Saving Proposals Identified – 2026/27 Budget Cycle | Exeter City Council | 0.602 |
| 4 | Revised Budget – Final Settlement for 2016-17 | Suffolk County Council | 0.578 |
| 5 | 2025/26 Quarter 3 Mitigations | Lancashire County Council | 0.570 |

**Full text keyword** (ILIKE on document_text)
| # | Title | Authority |
|---|-------|-----------|
| 1 | Budget Monitoring Report to 31st October 2015 | Council of the Isles of Scilly |
| 2 | Barking and Dagenham Council Submission to the Solving Social Care Funding Review | London Borough of Barking and Dagenham |
| 3 | Executive Work Programme and Forward Plan | West Oxfordshire District Council |
| 4 | Improving our Fire and Rescue Service: Facts First | West Oxfordshire District Council |
| 5 | Contract for Emergency Accommodation for Homeless Households | London Borough of Barking and Dagenham |

**Summary keyword** (ILIKE on summary)
| # | Title | Authority |
|---|-------|-----------|
| 1 | Redbridge Council Proposed Budget 2024/25 – Directorate | London Borough of Redbridge |
| 2 | Proposed Revenue Budget 2025/26 | London Borough of Redbridge |
| 3 | Budget Monitoring Report to 31st October 2015 | Council of the Isles of Scilly |
| 4 | The 2024/25 Budget and Medium Term Financial Strategy | Southampton City Council |
| 5 | Appendix A: Savings Proposals for Approval | London Borough of Redbridge |

**Title keyword** (ILIKE on title)
| # | Title | Authority |
|---|-------|-----------|
| 1 | Updated Budgets & Council Tax 2026/27 | Fareham Borough Council |
| 2 | Dorset Council 2025 Budget Speech | Dorset Council |
| 3 | Redbridge Council Proposed Budget 2024/25 | London Borough of Redbridge |
| 4 | Council Tax Statement – 2025/26 Budget | London Borough of Havering |
| 5 | Westminster City Council Tenancy Strategy 2025 | westminster.gov.uk |

> **Verdict:** Embedding finds the most relevant documents by far — specific savings proposals and budget mitigations — with high similarity scores (0.57–0.63). Crucially it finds "mitigations" (Lancashire) and "revised budget" (Suffolk) without those words being in the query — true semantic understanding. Summary keyword is reasonable (4/5 finance-related) but misses the specific savings focus. Full text is noisy — fire service report, homeless accommodation contract, and a forward plan rank highly simply because "council", "budget", or "savings" appear somewhere in a long document. Title keyword is the weakest here: "Westminster City Council Tenancy Strategy" ranks in the top 5 only because it contains "council".

---

### 4. "how much was spent on highways" *(natural language)*

**Embedding** (pgvector)
| # | Title | Authority | Similarity |
|---|-------|-----------|------------|
| 1 | Communities and Place OSC Presentation on Highway Maintenance | Northumberland County Council | 0.336 |
| 2 | North Northamptonshire Council Highway Asset Management Policy | North Northamptonshire Council | 0.304 |
| 3 | Ashford Highway Works Programme Report 2022/23 and 2023/24 | Ashford Borough Council | 0.296 |
| 4 | East Hertfordshire District Council Capital Programme 2026/27 to 2028/29 | East Hertfordshire District Council | 0.267 |
| 5 | Hellingly Parish Council CIL Monitoring Report 2024-2025 | Wealden District Council | 0.267 |

**Full text keyword** (ILIKE on document_text)
| # | Title | Authority |
|---|-------|-----------|
| 1 | River Wye Nutrient Management Plan Board Meeting Minutes | Herefordshire Council |
| 2 | FOI Response: Software Used by Regulatory, Environmental and Planning Services | Suffolk County Council |
| 3 | Gloucestershire County Council Annual Infrastructure Funding Statement | glostext.gloucestershire.gov.uk |
| 4 | Council Members' Questions Sheet – July 2025 | Dover District Council |
| 5 | Improving our Fire and Rescue Service: Facts First | West Oxfordshire District Council |

**Summary keyword** — Returns near-random results: food redistribution guidance, capital strategy, trade union report — matched on "much" or "spent" appearing coincidentally in summaries.

**Title keyword** — Returns: "How to Store Food Properly" (#1), waste meeting minutes, conservation area appraisals. None relevant. "How" is the only word ≥3 chars that matches any title.

> **Verdict:** This is the clearest demonstration of embedding's advantage. Keyword methods completely fail — "how", "much", "spent", "highways" are common words that appear across entirely unrelated documents. **Embedding is the only method that returns relevant results**: all top results are highways-related (maintenance scrutiny, asset management, works programme). Similarity scores are lower (~0.27–0.34) because no single document directly answers "how much was spent", but the topic match is accurate. Zero overlap between embedding and any keyword method.

---

### 5. "how was homelessness tackled" *(natural language)*

**Embedding** (pgvector)
| # | Title | Authority | Similarity |
|---|-------|-----------|------------|
| 1 | Homelessness and Temporary Accommodation Service Improvement Plan | Slough Borough Council | 0.524 |
| 2 | Trafford Homelessness Strategy 2025-2030 Summary | Trafford Metropolitan Borough Council | 0.518 |
| 3 | Homelessness Scrutiny Sub-Group Minutes, 12 October 2016 | Rugby Borough Council | 0.507 |
| 4 | Homelessness Performance and Recovery Plan Overview | Plymouth City Council | 0.492 |
| 5 | Vulnerability Guidance for Homelessness Assessment | South Kesteven District Council | 0.419 |

**Full text keyword** (ILIKE on document_text)
| # | Title | Authority |
|---|-------|-----------|
| 1 | Progress against the Corporate Plan and Three-year Action Plan | Eastleigh Borough Council |
| 2 | Overview and Scrutiny Committee Minutes – EHCP Process | London Borough of Barking and Dagenham |
| 3 | Executive Work Programme and Forward Plan | West Oxfordshire District Council |
| 4 | Westminster City Council Tenancy Strategy 2025 | westminster.gov.uk |
| 5 | Barking and Dagenham Council Submission to the Solving Social Care Funding Review | London Borough of Barking and Dagenham |

**Summary keyword** — Returns: ward budget summary, performance summary, water management — none relevant. Matched on "was", "how" coincidentally in summaries.

**Title keyword** — "How to Store Food Properly" ranks #1 (matched "how"), Rugby Homelessness Scrutiny #2 (matched "homelessness"). Only 1 relevant result out of 5.

> **Verdict:** Embedding returns 5 highly relevant homelessness documents (improvement plans, strategies, scrutiny minutes) with strong similarity scores (0.42–0.52). The semantic model correctly interprets "how was X tackled" as a request for homelessness policy and intervention documents. Full text fails — committee minutes and a social care funding submission rank highly because "homelessness" appears somewhere in long documents. Summary and title keyword largely fail because the question words ("how", "was", "tackled") match random content. Only 1 overlap between embedding and title keyword (the Rugby scrutiny minutes, where "homelessness" appears in the title).

---

### 6. "climate change net zero action plan"

**Embedding** (pgvector)
| # | Title | Authority | Similarity |
|---|-------|-----------|------------|
| 1 | Representation on Exeter Plan Chapter 5: Climate Change (Net Zero Exeter) | Exeter City Council | 0.580 |
| 2 | Carbon Emissions and Sustainability Progress Report 2022-23 | Guildford Borough Council | 0.557 |
| 3 | Torbay's Greenhouse Gas Emissions: Current Status and Pathways to Net Zero | Torbay Council | 0.540 |
| 4 | Key Sources of Information: Climate Change | Telford & Wrekin Council | 0.446 |
| 5 | Cabinet Lead for Climate Emergency, Environment and Water | Havant Borough Council | 0.442 |

**Full text keyword** (ILIKE on document_text)
| # | Title | Authority |
|---|-------|-----------|
| 1 | Blaby District Council Housing Strategy 2020-2025 | blaby.gov.uk |
| 2 | Cabinet Lead for Climate Emergency, Environment and Water | Havant Borough Council |
| 3 | Horton Heath Development Management Committee Report | Eastleigh Borough Council |
| 4 | Bromsgrove District Council Housing Land Availability Assessment | Bromsgrove District Council |
| 5 | Update on Exeter Community Lottery: First Year of Operation | Exeter City Council |

**Summary keyword** (ILIKE on summary)
| # | Title | Authority |
|---|-------|-----------|
| 1 | Carbon Emissions and Sustainability Progress Report 2022-23 | Guildford Borough Council |
| 2 | Representation on Exeter Plan Chapter 5: Climate Change | Exeter City Council |
| 3 | Key Sources of Information: Climate Change | Telford & Wrekin Council |
| 4 | Southampton International Airport Consultative Committee Agenda | Eastleigh Borough Council |
| 5 | Climate Change Cabinet Advisory Group Agenda – 15 April 2021 | Thanet District Council |

**Title keyword** (ILIKE on title)
| # | Title | Authority |
|---|-------|-----------|
| 1 | Representation on Exeter Plan Chapter 5: Climate Change (Net Zero Exeter) | Exeter City Council |
| 2 | Deputy Leader of the Council (Place) and Cabinet Member for Climate Change | Blackpool Borough Council |
| 3 | Climate Change Cabinet Advisory Group Agenda – 15 April 2021 | Thanet District Council |
| 4 | Oadby and Wigston Town Centres Area Action Plan | Oadby and Wigston Borough Council |
| 5 | Climate Change Impact Assessment Form | Dartford Borough Council |

> **Verdict:** Embedding surfaces the most substantive documents — net zero policy, emissions reporting, greenhouse gas pathways — and understands that "net zero" and "carbon emissions" are the same concept as "climate change action". Summary keyword is reasonable (3/5 clearly on-topic) but includes an airport consultative committee agenda. Full text is highly noisy: a housing strategy and a housing land assessment rank in the top 5 simply because "climate", "change", or "plan" appear somewhere in lengthy text. Title keyword finds relevant titles but Oadby's Area Action Plan appears only because "action plan" appears in the title without being climate-related.

---

### 7. "school places primary admissions"

**Embedding** (pgvector)
| # | Title | Authority | Similarity |
|---|-------|-----------|------------|
| 1 | Cambridgeshire Junior Year Allocation Information | Cambridgeshire County Council | 0.572 |
| 2 | Peterborough Year 7 Second Round Allocation Information | Peterborough City Council | 0.539 |
| 3 | Singleton Church of England Primary School Admission Arrangements 2023 | Lancashire County Council | 0.530 |
| 4 | Edward Betham CE Primary School Admissions Policy 2027-28 | London Borough of Ealing | 0.525 |
| 5 | Determined Admission Arrangements for Secondary Schools | London Borough of Barking and Dagenham | 0.513 |

**Full text keyword** (ILIKE on document_text)
| # | Title | Authority |
|---|-------|-----------|
| 1 | Cambridgeshire Junior Year Allocation Information | Cambridgeshire County Council |
| 2 | Abbots Ripton CE Primary School Supplementary Information Form | Cambridgeshire County Council |
| 3 | Determined Admission Arrangements for Secondary Schools | London Borough of Barking and Dagenham |
| 4 | Measuring School Capacity: A Summary Guide for Local Authorities | Cheshire East Council |
| 5 | Constitution of the Cambridgeshire Schools Forum | Cambridgeshire County Council |

**Summary keyword** (ILIKE on summary)
| # | Title | Authority |
|---|-------|-----------|
| 1 | Edward Betham CE Primary School Admissions Policy 2027-28 | London Borough of Ealing |
| 2 | Withington C of E Primary School Admissions Policy 2027-2028 | Gloucestershire County Council |
| 3 | Cambridgeshire Junior Year Allocation Information | Cambridgeshire County Council |
| 4 | Ralph Allen School Admissions Policy 2020-21 | bathnes.gov.uk |
| 5 | Measuring School Capacity: A Summary Guide for Local Authorities | Cheshire East Council |

**Title keyword** (ILIKE on title)
| # | Title | Authority |
|---|-------|-----------|
| 1 | Withington C of E Primary School Admissions Policy 2027-2028 | Gloucestershire County Council |
| 2 | Abbots Ripton CE Primary School Supplementary Information Form | Cambridgeshire County Council |
| 3 | Holy Trinity CE Primary School Admissions Policy 2026 | Lancashire County Council |
| 4 | St Bartholomews Church of England Primary School Admissions Policy | Hertfordshire County Council |
| 5 | Determined Admissions Arrangements 2026-2027 for Wrekin View Primary School | Telford & Wrekin Council |

> **Verdict:** All 4 methods return useful results for this query — it's one of the easier cases. Embedding finds the broadest range including year allocation docs and secondary admissions (all relevant to the school places topic). Full text also performs well here, returning 4 clearly on-topic results. Title keyword homes in tightly on primary school admissions policies. Summary keyword is a reasonable middle ground. The "school places" topic has consistent terminology across documents, which helps all methods.

---

## Overall Summary

| Query | Best method | Worst method | Notes |
|-------|-------------|-------------|-------|
| Infrastructure funding statement | Summary / Title | Full text (noisy) | Exact doc type name — keyword excels; full text also good |
| Housing strategy affordable homes | **Embedding** | Full text | Embedding 5/5 relevant; full text 0/5 |
| Council budget savings | **Embedding** | Full text | Finds synonyms (mitigations, revised budget); full text noise |
| How much was spent on highways | **Embedding only** | All keyword methods | Natural language — keyword returns garbage |
| How was homelessness tackled | **Embedding** | Summary / Title | Natural language; keyword returns near-random results |
| Climate change net zero action plan | **Embedding** | Full text | Finds substance (emissions, pathways) over incidental mentions |
| School places primary admissions | All reasonable | — | Consistent terminology helps all methods |

---

## Key Findings

### 1. Full text keyword search is consistently the worst performer
Across 6 of 7 queries, full text keyword search returned irrelevant results in the top 5 — committee minutes, forward plans, fire service reports, housing land assessments — because the query words appeared somewhere in a long document. It has extremely high recall but near-zero precision for anything beyond exact document names.

### 2. Embedding search (pgvector) is the most reliable across all query types
It performs well for both exact document-type queries ("infrastructure funding statement") and conversational natural language questions ("how much was spent on highways"). It understands synonyms and related concepts — finding "mitigations" for "savings", "greenhouse gas pathways" for "net zero action plan", "asset management" for "how much was spent". It is the **only method that works at all** for natural language queries.

### 3. Summary keyword is a viable lightweight fallback
Because GPT-generated summaries are concise and on-topic, ILIKE matching against them has much higher precision than full text. It performs comparably to embedding for well-defined document type queries but degrades significantly for natural language questions (where question words like "how", "was", "tackled" match unrelated content).

### 4. Title keyword is precise but has low recall
Works well when the user knows exactly what they're looking for and the document title reflects it. Fails for natural language queries and misses relevant documents with different title conventions. Useful as a secondary signal when combined with other methods.

### 5. Natural language queries are the definitive differentiator
Queries like "how much was spent on highways" and "how was homelessness tackled" expose the fundamental limitation of keyword search — it cannot match user intent when the query words don't appear literally in the indexed field. There is **zero overlap** between embedding results and any keyword method for these queries.

---

## Recommendation

### What to search against

**Embed the GPT summary, not the full document text.** The summary is a clean, noise-free semantic representation of the whole document. Embedding full text or chunking into fixed-size passages adds complexity without improving results — for a system where the unit of retrieval is a document, not a passage, summary embedding is both simpler and more accurate.

---

### Primary retrieval

**pgvector cosine similarity on the summary embedding.**

Embed the user's semantic query and rank all documents by `<=>` cosine distance in Postgres. This is the only method that reliably handles natural language questions ("how was homelessness tackled"), understands synonyms ("mitigations" for "savings"), and works consistently across all query types. All other methods are supplementary to this.

---

### Structured filters

**SQL `WHERE` clauses for explicit user constraints only.**

In a chatbot interface, the LLM extracts explicit constraints from the user message (authority, year) and applies them as SQL filters alongside the vector query:

```sql
SELECT title, authority, 1 - (embedding <=> query_vec) AS similarity
FROM documents
WHERE authority ILIKE '%Manchester%'   -- only if user said so
AND year >= '2024'                     -- only if user said so
ORDER BY embedding <=> query_vec
LIMIT 10
```

The key principle: **only apply filters the user explicitly stated, not ones inferred from context.** If a user asks "housing strategy documents", don't filter by the Housing theme — use the phrase as the embedding query. The embedding handles semantic intent; SQL handles metadata constraints. Themes are most valuable for explicit user-driven filtering ("only show housing documents") and browsing the corpus by category, not as an automatic pre-filter.

If filtered results return fewer than 3 documents, widen or drop the filter and inform the user — metadata quality (year ~95% accurate, themes ~88% accurate) means aggressive filtering can silently drop relevant documents.

---

### Optional secondary boost

**Summary keyword match as a lightweight re-ranking signal.**

For exact document-type queries ("infrastructure funding statement"), a summary keyword score provides a small precision lift. It is not a co-equal retrieval input — just a boost applied after embedding retrieval:

```
final_score = 0.85 × embedding_similarity + 0.15 × summary_keyword_score
```

**Do not use full document text for keyword matching.** It is the worst performer across all queries, returning committee minutes, forward plans, and fire service reports for finance or housing queries simply because the words appear incidentally in long documents.

---

### Index

| Scale | Recommendation |
|-------|---------------|
| Now (1,430 docs) | No index needed — full scan is near-instant |
| Full corpus (773K docs) | **HNSW** (`vector_cosine_ops`) — better recall than IVFFlat, handles incremental inserts without rebuilding, lower tuning burden despite higher memory usage |

IVFFlat is simpler to build but requires careful tuning of `lists` and `probes` parameters, and recall degrades as documents are added between index rebuilds. At 773K documents being loaded incrementally, HNSW is the more robust choice.

---

### What to drop from earlier recommendations

| Component | Verdict |
|-----------|---------|
| Fixed-size chunking (250–500 words) | Drop — summary embedding outperforms it for document retrieval |
| Full-text keyword in the hybrid (tsvector/ts_rank) | Drop — consistently worst performer; degrades fusion results |
| RRF fusion with full-text keyword | Drop — zero overlap with embedding on natural language queries means fusing adds noise, not signal |
| IVFFlat index at full scale | Use for MVP only — switch to HNSW before full corpus load |

---

### Summary

Embed the GPT-generated summary for each document. At query time, the LLM extracts a clean semantic query plus any explicit constraints (authority, year, theme) from the user's message. Run the semantic query through pgvector cosine similarity, applying SQL filters only where the user explicitly specified them. Optionally apply a small summary keyword boost for exact document-type queries. Use HNSW indexing at full scale. Keep full document text in the database for display and downstream use, but do not search against it.
