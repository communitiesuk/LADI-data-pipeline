# Search Method Evaluation Report

**Dataset:** 1,430 embedded UK local authority documents across 268 authorities
**Database:** Azure Postgres with pgvector (text-embedding-3-large, 3072 dimensions)
**Date:** June 2026

---

## Methods Tested

| Method | How it works | Notes |
|--------|-------------|-------|
| **Embedding** | Embed query → cosine similarity against stored vectors | Semantic understanding |
| **Full text** | ILIKE keyword match on full document text | High recall, noisy |
| **Summary** | ILIKE keyword match on GPT-generated summary | Concise, lower noise |
| **Title** | ILIKE keyword match on document title | Precise, low recall |

---

## Query Results

### 1. "infrastructure funding statement"

**Embedding**
| # | Title | Authority | Similarity |
|---|-------|-----------|------------|
| 1 | Gloucester City Council Annual Infrastructure Funding Statement 2024-25 | gloucester.moderngov.co.uk | 0.576 |
| 2 | Infrastructure Funding Statement – Priorities Report | sevenoaks.moderngov.co.uk | 0.527 |
| 3 | Gloucestershire County Council Annual Infrastructure Funding Statement 2023/24 | glostext.gloucestershire.gov.uk | 0.519 |
| 4 | Delivery and Monitoring of the Local Plan | Newcastle City Council | 0.470 |
| 5 | South Ribble Borough Council Annual Infrastructure Funding Statement 2024/25 | southribble.moderngov.co.uk | 0.446 |

**Full text keyword**
| # | Title | Authority |
|---|-------|-----------|
| 1 | Planning Committee Minutes – 23 September 2025 | LB Barking and Dagenham |
| 2 | Bath and North East Somerset Authorities Monitoring Report 2023/24 | bathnes.gov.uk |
| 3 | Adoption of Local Development Scheme (LDS) 2025 | Dover District Council |
| 4 | Horton Heath Development Management Committee Report | Eastleigh Borough Council |
| 5 | Worthing Investment Prospectus 2016 | adur-worthing.gov.uk |

**Summary keyword**
| # | Title | Authority |
|---|-------|-----------|
| 1 | Infrastructure Funding Statement – Priorities Report | sevenoaks.moderngov.co.uk |
| 2 | Gloucester City Council Annual IFS 2024-25 | gloucester.moderngov.co.uk |
| 3 | City of York Council IFS 2019-20 | york.gov.uk |
| 4 | South Ribble BC Annual IFS 2024/25 | southribble.moderngov.co.uk |
| 5 | Gloucestershire CC Annual IFS 2023/24 | glostext.gloucestershire.gov.uk |

**Title keyword**
| # | Title | Authority |
|---|-------|-----------|
| 1 | Gloucester City Council Annual IFS 2024-25 | gloucester.moderngov.co.uk |
| 2 | Infrastructure Funding Statement – Priorities Report | sevenoaks.moderngov.co.uk |
| 3 | South Ribble BC Annual IFS 2024/25 | southribble.moderngov.co.uk |
| 4 | City of York Council IFS 2019-20 | york.gov.uk |
| 5 | Gloucestershire CC Annual IFS 2023/24 | glostext.gloucestershire.gov.uk |

> **Verdict:** Summary and title keyword both return all 5 true IFS docs. Embedding returns 4 of 5 IFS docs but also surfaces a relevant Local Plan monitoring doc. Full text is entirely off — it picks up docs that merely mention "infrastructure" or "funding" incidentally (committee minutes, investment prospectus). **Summary and title search win here** because the query is an exact document type name.

---

### 2. "housing strategy affordable homes"

**Embedding**
| # | Title | Authority | Similarity |
|---|-------|-----------|------------|
| 1 | Housing Development Strategy 2025/26-2030/31 | Somerset Council | 0.583 |
| 2 | Blaby District Council Housing Strategy 2020-2025 | blaby.gov.uk | 0.583 |
| 3 | Accelerating Affordable Housing Delivery: A Quality Home for All | Stockton-on-Tees BC | 0.543 |
| 4 | Affordable Housing Development: Anticipated Delivery Schedule | Newark and Sherwood DC | 0.513 |
| 5 | Housing Position Statement – Cabinet Report | Devon County Council | 0.492 |

**Full text keyword**
| # | Title | Authority |
|---|-------|-----------|
| 1 | Horton Heath Development Management Committee Report | Eastleigh BC |
| 2 | Westminster City Council Tenancy Strategy 2025 | westminster.gov.uk |
| 3 | Adoption of Local Development Scheme (LDS) 2025 | Dover DC |
| 4 | Executive Work Programme and Forward Plan | West Oxfordshire DC |
| 5 | Planning Committee Minutes – 23 September 2025 | LB Barking and Dagenham |

**Summary keyword**
| # | Title | Authority |
|---|-------|-----------|
| 1 | Affordable Housing Development: Anticipated Delivery Schedule | Newark and Sherwood DC |
| 2 | Housing Position Statement – Cabinet Report | Devon County Council |
| 3 | Housing Development Strategy 2025/26-2030/31 | Somerset Council |
| 4 | Sustainability Appraisal Report (2012) | blaby.gov.uk |
| 5 | Developer's Main Planning Obligations for Barking Riverside | LB Barking and Dagenham |

**Title keyword**
| # | Title | Authority |
|---|-------|-----------|
| 1 | Support for Hyde Housing – 15 Social Rent Homes at Birdham | Chichester DC |
| 2 | Affordable Housing Development: Anticipated Delivery Schedule | Newark and Sherwood DC |
| 3 | Housing Development Strategy 2025/26-2030/31 | Somerset Council |
| 4 | Blaby District Council Housing Strategy 2020-2025 | blaby.gov.uk |
| 5 | Rugby Updated Housing Needs Evidence Report | Rugby BC |

> **Verdict:** Embedding performs best — all 5 results are directly relevant housing strategy documents. Summary keyword also does well (3/5 clearly on-topic) but surfaces a Sustainability Appraisal. Full text is again very noisy — committee minutes and a forward plan rank highly because "housing" appears in passing. Title keyword is good on precision but misses strategic documents that don't use "affordable" or "strategy" in their title.

---

### 3. "council budget savings"

**Embedding**
| # | Title | Authority | Similarity |
|---|-------|-----------|------------|
| 1 | Saving Proposals Identified – 2026/27 Budget Cycle | Exeter City Council | 0.633 |
| 2 | Appendix A: Savings Proposals for Approval | LB Redbridge | 0.618 |
| 3 | Saving Proposals Identified – 2026/27 Budget Cycle | Exeter City Council | 0.602 |
| 4 | Revised Budget – Final Settlement for 2016-17 | Suffolk County Council | 0.578 |
| 5 | 2025/26 Quarter 3 Mitigations | Lancashire County Council | 0.570 |

**Summary keyword**
| # | Title | Authority |
|---|-------|-----------|
| 1 | Redbridge Council Proposed Budget 2024/25 | LB Redbridge |
| 2 | Proposed Revenue Budget 2025/26 | LB Redbridge |
| 3 | Budget Monitoring Report to 31st October 2015 | Council of the Isles of Scilly |
| 4 | The 2024/25 Budget and Medium Term Financial Strategy | Southampton City Council |
| 5 | Appendix A: Savings Proposals for Approval | LB Redbridge |

> **Verdict:** Embedding and summary keyword both return strong finance documents. Embedding goes further by finding "mitigations" and budget revisions without those words being in the query — true semantic understanding. Full text returns irrelevant documents that mention "council" and "savings" in unrelated contexts.

---

### 4. "how much was spent on highways" *(natural language)*

**Embedding**
| # | Title | Authority | Similarity |
|---|-------|-----------|------------|
| 1 | Communities and Place OSC Presentation on Highway Maintenance | Northumberland CC | 0.336 |
| 2 | North Northamptonshire Council Highway Asset Management Policy | North Northamptonshire | 0.304 |
| 3 | Ashford Highway Works Programme Report 2022/23 and 2023/24 | Ashford BC | 0.296 |
| 4 | East Hertfordshire DC Capital Programme 2026/27 to 2028/29 | East Hertfordshire DC | 0.267 |
| 5 | Hellingly Parish Council CIL Monitoring Report | Wealden DC | 0.267 |

**Full text keyword**
| # | Title | Authority |
|---|-------|-----------|
| 1 | River Wye Nutrient Management Plan Board Meeting Minutes | Herefordshire Council |
| 2 | FOI Response: Software Used by Regulatory and Planning Services | Suffolk CC |
| 3 | Gloucestershire CC Annual Infrastructure Funding Statement | glostext.gloucestershire.gov.uk |
| 4 | Improving our Fire and Rescue Service: Facts First | West Oxfordshire DC |
| 5 | Council Members' Questions Sheet – July 2025 | Dover DC |

**Summary keyword** — Only 2 results, neither relevant (matched "much" or "spent" coincidentally)

**Title keyword** — Only 2 results: an equality impact assessment and a highways board minutes

> **Verdict:** This is a natural language question — keyword methods largely fail. Summary keyword returns near-random results. Full text happens to surface a highways doc (#3) but by accident. **Embedding clearly wins**: all top results are highways-related documents (maintenance, asset management, works programme). The similarity scores are lower (~0.30-0.34) reflecting that no single document directly answers "how much was spent" — but at least the results are on the right topic. This is the key advantage of semantic search for exploratory, question-style queries.

---

### 5. "how was homelessness tackled" *(natural language)*

**Embedding**
| # | Title | Authority | Similarity |
|---|-------|-----------|------------|
| 1 | Homelessness and Temporary Accommodation Service Improvement Plan | Slough BC | 0.524 |
| 2 | Trafford Homelessness Strategy 2025-2030 Summary | Trafford MBC | 0.518 |
| 3 | Homelessness Scrutiny Sub-Group Minutes | Rugby BC | 0.507 |
| 4 | Homelessness Performance and Recovery Plan Overview | Plymouth City Council | 0.492 |
| 5 | Vulnerability Guidance for Homelessness Assessment | South Kesteven DC | 0.419 |

**Full text keyword**
| # | Title | Authority |
|---|-------|-----------|
| 1 | Westminster City Council Tenancy Strategy 2025 | westminster.gov.uk |
| 2 | January 2021 Government Procurement Card Transactions | Dover DC |
| 3 | Grant Agreement for Citizens Advice Eastleigh | Eastleigh BC |
| 4 | Progress against Corporate Plan (2023-26) | Eastleigh BC |
| 5 | Executive Work Programme and Forward Plan | West Oxfordshire DC |

**Title keyword**
| # | Title | Authority |
|---|-------|-----------|
| 1 | Homelessness Scrutiny Sub-Group Minutes | Rugby BC |
| 2 | Homelessness and Temporary Accommodation Service Improvement Plan | Slough BC |
| 3 | Trafford Homelessness Strategy 2025-2030 Summary | Trafford MBC |
| 4 | Vulnerability Guidance for Homelessness Assessment | South Kesteven DC |
| 5 | Homelessness Performance and Recovery Plan Overview | Plymouth City Council |

> **Verdict:** Embedding and title keyword both return excellent, highly relevant results. Embedding scores are strong (0.42-0.52) because "homelessness tackled" maps cleanly to documents about homelessness strategies and improvement plans. Full text fails badly — procurement card transactions and a grant agreement rank highly simply because "homelessness" appears somewhere in the body text. Summary keyword also underperforms, returning a housing strategy and supplier payments. This query shows embedding handling a natural-language formulation ("how was X tackled") as well as exact title keyword.

---

### 6. "climate change net zero action plan"

**Embedding**
| # | Title | Authority | Similarity |
|---|-------|-----------|------------|
| 1 | Representation on Exeter Plan Chapter 5: Climate Change (Net Zero Exeter) | Exeter City Council | 0.580 |
| 2 | Carbon Emissions and Sustainability Progress Report 2022-23 | Guildford BC | 0.557 |
| 3 | Torbay's Greenhouse Gas Emissions: Current Status and Pathways to Net Zero | Torbay Council | 0.540 |
| 4 | Key Sources of Information: Climate Change | Telford & Wrekin | 0.446 |
| 5 | Cabinet Lead for Climate Emergency Update | Havant BC | 0.442 |

**Summary keyword**
| # | Title | Authority |
|---|-------|-----------|
| 1 | Carbon Emissions and Sustainability Progress Report 2022-23 | Guildford BC |
| 2 | Key Sources of Information: Climate Change | Telford & Wrekin |
| 3 | Southampton International Airport Consultative Committee Agenda | Eastleigh BC |
| 4 | Climate Change Impact Assessment Form | Dartford BC |
| 5 | Climate Change Topographic Sensitivity Maps | bathnes.gov.uk |

> **Verdict:** Embedding surfaces the most substantive documents — net zero policy, emissions reporting, greenhouse gas pathways. Summary keyword is reasonable (3/5 clearly on-topic) but includes a consultative committee agenda and a topographic sensitivity map. Full text is highly noisy, returning a housing strategy and a road construction report that happen to contain "climate" in passing.

---

### 7. "school places primary admissions"

**Embedding**
| # | Title | Authority | Similarity |
|---|-------|-----------|------------|
| 1 | Cambridgeshire Junior Year Allocation Information | Cambridgeshire CC | 0.572 |
| 2 | Peterborough Year 7 Second Round Allocation Information | Peterborough CC | 0.539 |
| 3 | Singleton CE Primary School Admission Arrangements 2023 | Lancashire CC | 0.530 |
| 4 | Edward Betham CE Primary School Admissions Policy 2027-28 | LB Ealing | 0.525 |
| 5 | Determined Admission Arrangements for Secondary Schools | LB Barking and Dagenham | 0.513 |

**Title keyword**
| # | Title | Authority |
|---|-------|-----------|
| 1 | Withington C of E Primary School Admissions Policy 2027-2028 | Gloucestershire CC |
| 2 | Abbots Ripton CE Primary School Supplementary Information Form | Cambridgeshire CC |
| 3 | Holy Trinity CE Primary School Admissions Policy 2026 | Lancashire CC |
| 4 | St Bartholomews CE Primary School Admissions Policy | Hertfordshire CC |
| 5 | Determined Admissions Arrangements for Wrekin View Primary School | Telford & Wrekin |

> **Verdict:** Both methods work well for this query. Embedding finds a slightly broader range including secondary allocation docs and year 7 information (relevant to the school places topic even if not strictly "primary"). Title keyword homes in specifically on primary school admissions policies. Both are usable; embedding has the edge for breadth.

---

## Overall Summary

| Query | Best method | Worst method | Notes |
|-------|-------------|-------------|-------|
| Infrastructure funding statement | Summary / Title | Full text | Exact doc type name — keyword excels |
| Housing strategy affordable homes | Embedding | Full text | Semantic range finds more relevant docs |
| Council budget savings | Embedding | Full text | Finds synonyms ("mitigations", "revision") |
| How much was spent on highways | **Embedding only** | Summary / Full text | Natural language — keyword nearly useless |
| How was homelessness tackled | Embedding / Title | Full text | Natural language handled well by embedding |
| Climate change net zero action plan | Embedding | Full text | Finds substance over incidental mentions |
| School places primary admissions | Embedding / Title | Full text | Both work; embedding slightly broader |

---

## Key Findings

### 1. Full text keyword search is consistently the worst performer
Across all 7 queries, full text search returned irrelevant results in the top 5 — committee minutes, procurement card transactions, forward plans — simply because the query words appeared somewhere in a long document. It has high recall but very poor precision.

### 2. Embedding search is the most reliable across all query types
It performs well for both exact document-type queries ("infrastructure funding statement") and natural language questions ("how much was spent on highways"). It understands synonyms and semantics — finding "mitigations" for "savings", "greenhouse gas pathways" for "net zero action plan".

### 3. Summary keyword is a strong lightweight alternative
Because GPT-generated summaries are concise and on-topic, keyword matching against them has much higher precision than full text. It performs comparably to embedding for well-defined document type queries but degrades for natural language questions.

### 4. Title keyword is precise but has low recall
Works well when the user knows exactly what they're looking for (IFS, housing strategy, school admissions). Fails for natural language queries and misses relevant documents with different title conventions.

### 5. Natural language queries are the key differentiator
Queries like "how much was spent on highways" and "how was homelessness tackled" reveal the fundamental limitation of keyword search — it cannot match user intent when the query words don't appear literally in the document. Embedding handles these correctly.

---

## Recommendation

For the LADI search system:

**Primary: Embedding search** — best overall performance, handles both structured and natural language queries, works against the database via pgvector `<=>` operator.

**Secondary: Summary keyword** — useful as a no-API fallback or for filtering/faceting by known terms (e.g. filtering by authority, year, or theme alongside embedding search).

**Avoid: Full text keyword as a primary search method** — too noisy given the volume and variety of LA documents. Could be retained for exact-phrase matching on specific known document identifiers (e.g. a specific reference number).

A hybrid approach — embedding search ranked by cosine similarity, with summary keyword used as a tie-breaker or boosting signal — would give the strongest results in production.
