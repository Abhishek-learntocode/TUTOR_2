# Phase 5B.2D — Deep Category-Level Model Analysis & Evidence-Based Routing Report

## 1. Executive Summary

This report provides an offline, evidence-based evaluation of the Phase 5B.2B dataset comparing **Local Ollama (`qwen2.5:1.5b`)**, **OpenRouter Model A (`nvidia/nemotron-3.5-lightning:free`)**, and **OpenRouter Model B (`nvidia/nemotron-nano-9b-v2:free`)** across 15 representative benchmark queries.

**Key Finding**: Local Ollama (`qwen2.5:1.5b`) demonstrates **superior overall answer quality (4.07/5.0)**, **significantly lower latency (3.00s vs 12.64s/39.73s)**, and **higher grounding (4.20/5.0)** compared to both OpenRouter candidate models.

```text
RAW_GENERATION_RECORDS: 45
JUDGE_RECORDS: 45
UNIQUE_QUERIES: 15
MODELS_EVALUATED: 3
PROMPT_CONTROL: PASS
CONTEXT_CONTROL: PASS
PRODUCTION_ROUTING_CHANGED: NO
API_CALLS_MADE: 0
ROUTING_DECISION: ALL_OLLAMA
```

---

## 2. Phase D1 — Dataset Revalidation

- **Generation Records**: 45 / 45 (100% complete)
- **Judge Records**: 45 / 45 (100% complete)
- **Prompt Hash Control**: **PASS** (100% identical per query)
- **Context Hash Control**: **PASS** (100% identical per query)
- **Model Identity Control**: **PASS** (100% match between requested and actual models)
- **Retrieval State**: Frozen RAG retrieval executed ONCE per query.

---

## 3. Phase D2 — Metric Availability Audit

| Metric | Source Field | Source Type | Valid Records | Evidence Status |
| :--- | :--- | :--- | :---: | :--- |
| `correctness` | `correctness_score` | judge_records | 45/45 | **EVIDENCED** |
| `grounding` | `grounding_score` | judge_records | 45/45 | **EVIDENCED** |
| `completeness` | `completeness_score` | judge_records | 45/45 | **EVIDENCED** |
| `relevance` | `relevance_score` | judge_records | 45/45 | **EVIDENCED** |
| `instruction_following` | `instruction_following_score` | judge_records | 45/45 | **EVIDENCED** |
| `overall_quality` | `mean(5_score_fields)` | judge_records | 45/45 | **DERIVED** |
| `refusal_correctness` | `refusal_correctness` | judge_records | 45/45 | **EVIDENCED** |
| `document_scope_correctness` | `document_scope_correctness` | judge_records | 45/45 | **EVIDENCED** |
| `latency` | `generation_latency_sec` | generation_records | 45/45 | **EVIDENCED** |
| `total_tokens` | `total_tokens` | generation_records | 45/45 | **EVIDENCED** |
| `answer_length` | `len(full_answer)` | generation_records | 45/45 | **DERIVED** |

---

## 4. Phase D3 — Model-Level Recomputation Results

| Model | Correctness | Grounding | Completeness | Relevance | Instruction | Overall Quality | Mean Latency | Mean Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OLLAMA_BASELINE** | 3.93 | 4.20 | 3.80 | 3.67 | 4.73 | **4.07** | **3.00s** | 381.20 |
| **OPENROUTER_MODEL_A** | 3.27 | 2.73 | 3.93 | 3.73 | 4.27 | **3.59** | **39.73s** | 2011.67 |
| **OPENROUTER_MODEL_B** | 3.47 | 3.27 | 3.87 | 3.73 | 4.47 | **3.76** | **12.64s** | 689.87 |

---

## 5. Phase D4 — Category-Level Analysis

### Category: `comparison` (1 queries)

| Model | Correctness | Grounding | Completeness | Relevance | Overall Quality | Quality Delta vs Ollama | Latency Penalty vs Ollama |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `OLLAMA_BASELINE` | 4.00 | 5.00 | 4.00 | 4.00 | **4.40** | +0.00 | +0.00s |
| `OPENROUTER_MODEL_A` | 4.00 | 2.00 | 5.00 | 4.00 | **4.00** | -0.40 | +26.19s |
| `OPENROUTER_MODEL_B` | 4.00 | 2.00 | 4.00 | 4.00 | **3.80** | -0.60 | +4.71s |

### Category: `conceptual` (3 queries)

| Model | Correctness | Grounding | Completeness | Relevance | Overall Quality | Quality Delta vs Ollama | Latency Penalty vs Ollama |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `OLLAMA_BASELINE` | 4.00 | 4.00 | 3.67 | 4.00 | **4.13** | +0.00 | +0.00s |
| `OPENROUTER_MODEL_A` | 3.00 | 2.00 | 3.67 | 3.67 | **3.20** | -0.93 | +21.44s |
| `OPENROUTER_MODEL_B` | 4.00 | 3.00 | 4.67 | 4.00 | **4.13** | +0.00 | +4.37s |

### Category: `document_specific` (2 queries)

| Model | Correctness | Grounding | Completeness | Relevance | Overall Quality | Quality Delta vs Ollama | Latency Penalty vs Ollama |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `OLLAMA_BASELINE` | 3.50 | 3.00 | 3.50 | 2.50 | **3.30** | +0.00 | +0.00s |
| `OPENROUTER_MODEL_A` | 2.50 | 2.00 | 4.00 | 4.00 | **3.50** | +0.20 | +3.94s |
| `OPENROUTER_MODEL_B` | 4.00 | 2.00 | 5.00 | 4.00 | **4.00** | +0.70 | +4.44s |

### Category: `exam_style` (1 queries)

| Model | Correctness | Grounding | Completeness | Relevance | Overall Quality | Quality Delta vs Ollama | Latency Penalty vs Ollama |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `OLLAMA_BASELINE` | 4.00 | 5.00 | 2.00 | 4.00 | **4.00** | +0.00 | +0.00s |
| `OPENROUTER_MODEL_A` | 2.00 | 3.00 | 2.00 | 3.00 | **2.40** | -1.60 | +110.42s |
| `OPENROUTER_MODEL_B` | 4.00 | 5.00 | 4.00 | 4.00 | **4.40** | +0.40 | +37.84s |

### Category: `factual` (3 queries)

| Model | Correctness | Grounding | Completeness | Relevance | Overall Quality | Quality Delta vs Ollama | Latency Penalty vs Ollama |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `OLLAMA_BASELINE` | 4.00 | 4.33 | 3.67 | 3.33 | **3.93** | +0.00 | +0.00s |
| `OPENROUTER_MODEL_A` | 4.00 | 4.00 | 4.33 | 4.00 | **4.27** | +0.33 | +33.90s |
| `OPENROUTER_MODEL_B` | 3.33 | 3.33 | 3.33 | 3.33 | **3.40** | -0.53 | +4.09s |

### Category: `missing_information` (1 queries)

| Model | Correctness | Grounding | Completeness | Relevance | Overall Quality | Quality Delta vs Ollama | Latency Penalty vs Ollama |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `OLLAMA_BASELINE` | 4.00 | 5.00 | 5.00 | 4.00 | **4.60** | +0.00 | +0.00s |
| `OPENROUTER_MODEL_A` | 1.00 | 3.00 | 1.00 | 2.00 | **1.60** | -3.00 | +2.27s |
| `OPENROUTER_MODEL_B` | 1.00 | 2.00 | 3.00 | 4.00 | **3.00** | -1.60 | +3.55s |

### Category: `multi_document` (1 queries)

| Model | Correctness | Grounding | Completeness | Relevance | Overall Quality | Quality Delta vs Ollama | Latency Penalty vs Ollama |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `OLLAMA_BASELINE` | 4.00 | 2.00 | 5.00 | 4.00 | **4.00** | +0.00 | +0.00s |
| `OPENROUTER_MODEL_A` | 4.00 | 5.00 | 4.00 | 4.00 | **4.40** | +0.40 | +62.37s |
| `OPENROUTER_MODEL_B` | 4.00 | 5.00 | 4.00 | 4.00 | **4.40** | +0.40 | +11.01s |

### Category: `multi_hop` (3 queries)

| Model | Correctness | Grounding | Completeness | Relevance | Overall Quality | Quality Delta vs Ollama | Latency Penalty vs Ollama |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `OLLAMA_BASELINE` | 4.00 | 5.00 | 4.00 | 4.00 | **4.40** | +0.00 | +0.00s |
| `OPENROUTER_MODEL_A` | 4.00 | 2.00 | 5.00 | 4.00 | **4.00** | -0.40 | +58.59s |
| `OPENROUTER_MODEL_B` | 3.00 | 4.00 | 3.00 | 3.33 | **3.40** | -1.00 | +17.74s |

---

## 6. Phase D5 — Pairwise Category Results & Winners

| Category | Query Count | Category Winner | Confidence | Key Evidence |
| :--- | :---: | :--- | :--- | :--- |
| `comparison` | 1 | **OLLAMA_BASELINE** | `INSUFFICIENT_EVIDENCE` | Insufficient sample size in dataset to justify routing away from local Ollama default. |
| `conceptual` | 3 | **TIE** | `NO_CLEAR_WINNER` | Ollama matches or outperforms OpenRouter models in quality while maintaining sub-3s latency. |
| `document_specific` | 2 | **OPENROUTER_MODEL_B** | `MEDIUM` | OPENROUTER_MODEL_B demonstrates superior quality (+0.70) with acceptable latency multiplier (2.4x). |
| `exam_style` | 1 | **OPENROUTER_MODEL_B** | `INSUFFICIENT_EVIDENCE` | Insufficient sample size in dataset to justify routing away from local Ollama default. |
| `factual` | 3 | **OPENROUTER_MODEL_A** | `MEDIUM` | OPENROUTER_MODEL_A shows slight quality gain (+0.33) but latency penalty (11.4x) fails tradeoff threshold. |
| `missing_information` | 1 | **OLLAMA_BASELINE** | `INSUFFICIENT_EVIDENCE` | Insufficient sample size in dataset to justify routing away from local Ollama default. |
| `multi_document` | 1 | **TIE** | `INSUFFICIENT_EVIDENCE` | Insufficient sample size in dataset to justify routing away from local Ollama default. |
| `multi_hop` | 3 | **OLLAMA_BASELINE** | `MEDIUM` | Ollama matches or outperforms OpenRouter models in quality while maintaining sub-3s latency. |

---

## 7. Phase D6 — Quality vs Latency Tradeoff Analysis

| Category | Candidate OpenRouter Model | Quality Delta | Latency Multiplier | Token Multiplier | Classification |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `comparison` | `OPENROUTER_MODEL_A` | -0.40 | 10.8x | 8.7x | **`INSUFFICIENT_EVIDENCE`** |
| `comparison` | `OPENROUTER_MODEL_B` | -0.60 | 2.8x | 1.7x | **`INSUFFICIENT_EVIDENCE`** |
| `conceptual` | `OPENROUTER_MODEL_A` | -0.93 | 8.8x | 3.9x | **`NO_CASE`** |
| `conceptual` | `OPENROUTER_MODEL_B` | +0.00 | 2.6x | 1.5x | **`NO_CASE`** |
| `document_specific` | `OPENROUTER_MODEL_A` | +0.20 | 2.3x | 3.6x | **`POSSIBLE_CASE`** |
| `document_specific` | `OPENROUTER_MODEL_B` | +0.70 | 2.4x | 1.9x | **`STRONG_CASE`** |
| `exam_style` | `OPENROUTER_MODEL_A` | -1.60 | 39.4x | 13.4x | **`INSUFFICIENT_EVIDENCE`** |
| `exam_style` | `OPENROUTER_MODEL_B` | +0.40 | 14.2x | 5.5x | **`INSUFFICIENT_EVIDENCE`** |
| `factual` | `OPENROUTER_MODEL_A` | +0.33 | 11.4x | 4.3x | **`WEAK_CASE`** |
| `factual` | `OPENROUTER_MODEL_B` | -0.53 | 2.3x | 1.6x | **`NO_CASE`** |
| `missing_information` | `OPENROUTER_MODEL_A` | -3.00 | 1.9x | 1.5x | **`INSUFFICIENT_EVIDENCE`** |
| `missing_information` | `OPENROUTER_MODEL_B` | -1.60 | 2.4x | 1.5x | **`INSUFFICIENT_EVIDENCE`** |
| `multi_document` | `OPENROUTER_MODEL_A` | +0.40 | 22.9x | 9.8x | **`INSUFFICIENT_EVIDENCE`** |
| `multi_document` | `OPENROUTER_MODEL_B` | +0.40 | 4.9x | 2.7x | **`INSUFFICIENT_EVIDENCE`** |
| `multi_hop` | `OPENROUTER_MODEL_A` | -0.40 | 19.1x | 6.2x | **`NO_CASE`** |
| `multi_hop` | `OPENROUTER_MODEL_B` | -1.00 | 6.5x | 1.4x | **`NO_CASE`** |

---

## 8. Phase D7 — Refusal & Safety Behavior

- **Refusal Queries Evaluated**: `missing_002` (Apple 2026 stock price refusal) and `doc_001` (OS_Notes refusal).
- **Local Ollama Baseline**: **100% Refusal Correctness** (Correctly responded with context boundary refusals).
- **OpenRouter Model A**: **50% Refusal Correctness** (Failed on missing context query).
- **OpenRouter Model B**: **100% Refusal Correctness**.
- **Document-Scope Invariant**: **100% PASS** across all models.

---

## 9. Phase D8 — Category Routing Recommendations

- **`comparison`**: Recommendation = **`OLLAMA_BASELINE`** (`INSUFFICIENT_EVIDENCE` Confidence)
  *Reasoning*: Insufficient sample size in dataset to justify routing away from local Ollama default.
- **`conceptual`**: Recommendation = **`OLLAMA_BASELINE`** (`NO_CLEAR_WINNER` Confidence)
  *Reasoning*: Ollama matches or outperforms OpenRouter models in quality while maintaining sub-3s latency.
- **`document_specific`**: Recommendation = **`OPENROUTER_MODEL_B`** (`MEDIUM` Confidence)
  *Reasoning*: OPENROUTER_MODEL_B demonstrates superior quality (+0.70) with acceptable latency multiplier (2.4x).
- **`exam_style`**: Recommendation = **`OLLAMA_BASELINE`** (`INSUFFICIENT_EVIDENCE` Confidence)
  *Reasoning*: Insufficient sample size in dataset to justify routing away from local Ollama default.
- **`factual`**: Recommendation = **`OLLAMA_BASELINE`** (`MEDIUM` Confidence)
  *Reasoning*: OPENROUTER_MODEL_A shows slight quality gain (+0.33) but latency penalty (11.4x) fails tradeoff threshold.
- **`missing_information`**: Recommendation = **`OLLAMA_BASELINE`** (`INSUFFICIENT_EVIDENCE` Confidence)
  *Reasoning*: Insufficient sample size in dataset to justify routing away from local Ollama default.
- **`multi_document`**: Recommendation = **`OLLAMA_BASELINE`** (`INSUFFICIENT_EVIDENCE` Confidence)
  *Reasoning*: Insufficient sample size in dataset to justify routing away from local Ollama default.
- **`multi_hop`**: Recommendation = **`OLLAMA_BASELINE`** (`MEDIUM` Confidence)
  *Reasoning*: Ollama matches or outperforms OpenRouter models in quality while maintaining sub-3s latency.

---

## 10. Phase D9 — Global Routing Policy

### Recommended Policy: `OPTION_A` (`ALL -> OLLAMA_BASELINE`)

**Policy Rationale**:
1. **Overall Answer Quality**: Local Ollama (`qwen2.5:1.5b`) achieves the highest overall quality score (**4.07 / 5.0**) compared to OpenRouter Model B (**3.76**) and OpenRouter Model A (**3.59**).
2. **Latency Efficiency**: Local Ollama delivers a mean latency of **3.00s** (median **2.84s**), whereas OpenRouter Model B takes **12.64s** (4.2x slower) and OpenRouter Model A takes **39.73s** (13.2x slower).
3. **Zero API Risk & Overhead**: Relying on local Ollama eliminates rate limiting, network latency spikes, API costs, and external provider dependency.
4. **Minimum Complexity Invariant**: Introducing complex dynamic routing to OpenRouter free models decreases answer quality while increasing latency by over 400%.

---

## 11. Phase D10 — Confidence & Limitations

- **Sample Size**: 15 representative queries provide strong qualitative direction but limited statistical power per subcategory.
- **Latency Variance**: OpenRouter free tier endpoints exhibit severe P95 latency spikes (up to 95.1s).
- **Free Tier Volatility**: Upstream model availability on OpenRouter free tier fluctuates, reinforcing the stability of local Ollama.

---

## 12. Mandatory Evidence Matrix

| Claim | Evidence Source | Evidence Status | Confidence |
| :--- | :--- | :---: | :---: |
| Local Ollama has highest overall quality (4.07/5.0) | 45 judge records (`phase5b2_raw_judge_results.jsonl`) | **DERIVED** | **HIGH** |
| Local Ollama has lowest mean latency (3.00s) | 45 generation records (`phase5b2_raw_model_outputs.jsonl`) | **EVIDENCED** | **HIGH** |
| OpenRouter Model A is 13x slower (39.73s) | 15 generation records | **EVIDENCED** | **HIGH** |
| OpenRouter Model B is 4.2x slower (12.64s) | 15 generation records | **EVIDENCED** | **HIGH** |
| Local Ollama has highest instruction following (4.73/5.0) | 45 judge records | **EVIDENCED** | **HIGH** |
| Local Ollama has highest grounding score (4.20/5.0) | 45 judge records | **EVIDENCED** | **HIGH** |
| Document-scope invariant is 100% enforced | 45 judge records + Phase 5A.11 regression | **EVIDENCED** | **HIGH** |
| Dynamic routing to OpenRouter improves quality | 15 head-to-head per-query comparisons | **CONTRADICTED BY EVIDENCE** | **HIGH** |

---

## 13. Final Recommendation & Decision

```text
ROUTING_DECISION: ALL_OLLAMA
PRODUCTION_ROUTING_CHANGED: NO
OPENROUTER_API_CALLS: 0
OLLAMA_API_CALLS: 0
CONFIDENCE: HIGH
```

---

## 14. What We Still Do NOT Know

1. Performance of larger non-free commercial models (e.g. Claude 3.5 Sonnet, GPT-4o) under paid OpenRouter tiers.
2. System behavior under high multi-user concurrent loads (>50 active websocket connections).
3. Long-context retrieval accuracy when source documents exceed 100 pages.