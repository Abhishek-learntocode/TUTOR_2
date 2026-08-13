# Phase 5B.2C — Model Quality, Reliability & Routing Analysis

## 1. Executive Summary

This report analyzes the frozen Phase 5B.2B dataset. No new model/API calls were performed.

```text
RAW_GENERATION_RECORDS: 45
JUDGE_RECORDS: 45
UNIQUE_QUERIES: 15
MODELS: 3
PROMPT_CONTROL: PASS
CONTEXT_CONTROL: PASS
```

## 2. Model-Level Quality Comparison

| Model | Correctness | Grounding | Completeness | Relevance | Instruction | Overall | Latency | Tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLLAMA_BASELINE | 3.93 | 4.20 | 3.80 | 3.67 | 4.73 | 4.07 | 3.00s | 381.20 |
| OPENROUTER_MODEL_A | 3.27 | 2.73 | 3.93 | 3.73 | 4.27 | 3.59 | 39.73s | 2011.67 |
| OPENROUTER_MODEL_B | 3.47 | 3.27 | 3.87 | 3.73 | 4.47 | 3.76 | 12.64s | 689.87 |

## 3. Category-Level Analysis

### comparison

| Model | Correctness | Grounding | Completeness | Relevance | Instruction | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OLLAMA_BASELINE | 4.00 | 5.00 | 4.00 | 4.00 | 5.00 | 4.40 |
| OPENROUTER_MODEL_A | 4.00 | 2.00 | 5.00 | 4.00 | 5.00 | 4.00 |
| OPENROUTER_MODEL_B | 4.00 | 2.00 | 4.00 | 4.00 | 5.00 | 3.80 |

### conceptual

| Model | Correctness | Grounding | Completeness | Relevance | Instruction | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OLLAMA_BASELINE | 4.00 | 4.00 | 3.67 | 4.00 | 5.00 | 4.13 |
| OPENROUTER_MODEL_A | 3.00 | 2.00 | 3.67 | 3.67 | 3.67 | 3.20 |
| OPENROUTER_MODEL_B | 4.00 | 3.00 | 4.67 | 4.00 | 5.00 | 4.13 |

### document_specific

| Model | Correctness | Grounding | Completeness | Relevance | Instruction | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OLLAMA_BASELINE | 3.50 | 3.00 | 3.50 | 2.50 | 4.00 | 3.30 |
| OPENROUTER_MODEL_A | 2.50 | 2.00 | 4.00 | 4.00 | 5.00 | 3.50 |
| OPENROUTER_MODEL_B | 4.00 | 2.00 | 5.00 | 4.00 | 5.00 | 4.00 |

### exam_style

| Model | Correctness | Grounding | Completeness | Relevance | Instruction | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OLLAMA_BASELINE | 4.00 | 5.00 | 2.00 | 4.00 | 5.00 | 4.00 |
| OPENROUTER_MODEL_A | 2.00 | 3.00 | 2.00 | 3.00 | 2.00 | 2.40 |
| OPENROUTER_MODEL_B | 4.00 | 5.00 | 4.00 | 4.00 | 5.00 | 4.40 |

### factual

| Model | Correctness | Grounding | Completeness | Relevance | Instruction | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OLLAMA_BASELINE | 4.00 | 4.33 | 3.67 | 3.33 | 4.33 | 3.93 |
| OPENROUTER_MODEL_A | 4.00 | 4.00 | 4.33 | 4.00 | 5.00 | 4.27 |
| OPENROUTER_MODEL_B | 3.33 | 3.33 | 3.33 | 3.33 | 3.67 | 3.40 |

### missing_information

| Model | Correctness | Grounding | Completeness | Relevance | Instruction | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OLLAMA_BASELINE | 4.00 | 5.00 | 5.00 | 4.00 | 5.00 | 4.60 |
| OPENROUTER_MODEL_A | 1.00 | 3.00 | 1.00 | 2.00 | 1.00 | 1.60 |
| OPENROUTER_MODEL_B | 1.00 | 2.00 | 3.00 | 4.00 | 5.00 | 3.00 |

### multi_document

| Model | Correctness | Grounding | Completeness | Relevance | Instruction | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OLLAMA_BASELINE | 4.00 | 2.00 | 5.00 | 4.00 | 5.00 | 4.00 |
| OPENROUTER_MODEL_A | 4.00 | 5.00 | 4.00 | 4.00 | 5.00 | 4.40 |
| OPENROUTER_MODEL_B | 4.00 | 5.00 | 4.00 | 4.00 | 5.00 | 4.40 |

### multi_hop

| Model | Correctness | Grounding | Completeness | Relevance | Instruction | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OLLAMA_BASELINE | 4.00 | 5.00 | 4.00 | 4.00 | 5.00 | 4.40 |
| OPENROUTER_MODEL_A | 4.00 | 2.00 | 5.00 | 4.00 | 5.00 | 4.00 |
| OPENROUTER_MODEL_B | 3.00 | 4.00 | 3.00 | 3.33 | 3.67 | 3.40 |

## 4. Pairwise Model Comparison

### OLLAMA_BASELINE_VS_OPENROUTER_MODEL_A

- Comparisons: 15
- OLLAMA_BASELINE wins: 9
- OPENROUTER_MODEL_A wins: 3
- Ties: 3
- OLLAMA_BASELINE win rate: 60.00%
- OPENROUTER_MODEL_A win rate: 20.00%
- Mean score delta: 0.48

### OLLAMA_BASELINE_VS_OPENROUTER_MODEL_B

- Comparisons: 15
- OLLAMA_BASELINE wins: 5
- OPENROUTER_MODEL_B wins: 3
- Ties: 7
- OLLAMA_BASELINE win rate: 33.33%
- OPENROUTER_MODEL_B win rate: 20.00%
- Mean score delta: 0.31

### OPENROUTER_MODEL_A_VS_OPENROUTER_MODEL_B

- Comparisons: 15
- OPENROUTER_MODEL_A wins: 3
- OPENROUTER_MODEL_B wins: 7
- Ties: 5
- OPENROUTER_MODEL_A win rate: 20.00%
- OPENROUTER_MODEL_B win rate: 46.67%
- Mean score delta: -0.17

## 5. Per-Query Winner Matrix

| Query | Category | OLLAMA_BASELINE | OPENROUTER_MODEL_A | OPENROUTER_MODEL_B | Winner |
| --- | --- | --- | --- | --- | --- |
| compare_001 | comparison | 4.40 | 4.00 | 3.80 | OLLAMA_BASELINE |
| concept_001 | conceptual | 4.40 | 4.00 | 4.40 | TIE |
| concept_002 | conceptual | 4.00 | 4.00 | 4.00 | TIE |
| concept_003 | conceptual | 4.00 | 1.60 | 4.00 | TIE |
| doc_001 | document_specific | 4.00 | 4.00 | 4.00 | TIE |
| doc_003 | document_specific | 2.60 | 3.00 | 4.00 | OPENROUTER_MODEL_B |
| exam_001 | exam_style | 4.00 | 2.40 | 4.40 | OPENROUTER_MODEL_B |
| factual_001 | factual | 3.00 | 4.40 | 1.80 | OPENROUTER_MODEL_A |
| factual_002 | factual | 4.40 | 4.40 | 4.40 | TIE |
| factual_003 | factual | 4.40 | 4.00 | 4.00 | OLLAMA_BASELINE |
| missing_002 | missing_information | 4.60 | 1.60 | 3.00 | OLLAMA_BASELINE |
| multidoc_001 | multi_document | 4.00 | 4.40 | 4.40 | TIE |
| multihop_001 | multi_hop | 4.40 | 4.00 | 4.40 | TIE |
| multihop_002 | multi_hop | 4.40 | 4.00 | 4.40 | TIE |
| multihop_004 | multi_hop | 4.40 | 4.00 | 1.40 | OLLAMA_BASELINE |

## 6. Latency Analysis

| Model | Mean | Median | P95 |
| --- | ---: | ---: | ---: |
| OLLAMA_BASELINE | 3.00s | 2.84s | 4.06s |
| OPENROUTER_MODEL_A | 39.73s | 29.36s | 95.16s |
| OPENROUTER_MODEL_B | 12.64s | 7.67s | 41.54s |

## 7. Token Usage

| Model | Mean Total Tokens | Median Total Tokens |
| --- | ---: | ---: |
| OLLAMA_BASELINE | 381.20 | 399.00 |
| OPENROUTER_MODEL_A | 2011.67 | 1607.00 |
| OPENROUTER_MODEL_B | 689.87 | 657.00 |

## 8. Experimental Controls

- Prompt hash control: **PASS**
- Context hash control: **PASS**
- Retrieval was frozen before model comparison.
- No production routing changes were made.
- No additional OpenRouter requests were made during analysis.

## 9. Routing-Oriented Observations

### Quality ranking

1. `OLLAMA_BASELINE` — overall judge score 4.07/5
2. `OPENROUTER_MODEL_B` — overall judge score 3.76/5
3. `OPENROUTER_MODEL_A` — overall judge score 3.59/5

### Latency ranking

1. `OLLAMA_BASELINE` — 3.00s mean
2. `OPENROUTER_MODEL_B` — 12.64s mean
3. `OPENROUTER_MODEL_A` — 39.73s mean

### Important interpretation

Quality scores and latency must be considered jointly. A higher quality score does not automatically justify routing every query to that model.

This report does not change production routing.

## 10. Final Status

```text
PHASE_5B2C_STATUS: COMPLETE
DATASET_INTEGRITY: PASS
JUDGE_SCHEMA_DETECTED: TOP_LEVEL_SCORE_FIELDS
JUDGE_SCORE_EXTRACTION: PASS
LATENCY_EXTRACTION: PASS
PROMPT_CONTROL: PASS
CONTEXT_CONTROL: PASS
MODEL_IDENTITY_CONTROL: PASS
PAIRWISE_ANALYSIS: PASS
API_CALLS_DURING_ANALYSIS: 0
PRODUCTION_FILES_MODIFIED: 0
MODEL_SELECTION_DECISION: DEFERRED
```
