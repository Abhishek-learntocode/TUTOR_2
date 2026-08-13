# Performance Phase 5B.2B — Controlled Model Evaluation Data Collection Report

## 1. Executive Summary
Phase 5B.2B successfully collected a clean, reproducible, un-truncated raw dataset for comparing local Ollama (`qwen2.5:1.5b`) against 2 fixed free OpenRouter models (`nvidia/nemotron-3.5-lightning:free` and `nvidia/nemotron-nano-9b-v2:free`) across 15 representative queries.
RAG retrieval was executed ONCE per query and frozen. `prompt_sha256` and `context_sha256` were verified to be 100% identical across all compared models per query.
**Note**: Zero production code changes were made during this phase (`PRODUCTION_FILES_MODIFIED: 0`). Final model-selection decision is **DEFERRED** to Phase 5B.2C.

## 2. OpenRouter API Budget Summary

| Metric | Value | Budget Limit | Status |
| :--- | :--- | :--- | :--- |
| **Hard API Request Limit** | 40 | 40 | PASS |
| **OpenRouter Requests Used** | 30 | <= 40 | PASS |
| **OpenRouter Requests Remaining** | 10 | Buffer | PASS |
| **Successful Requests** | 30 | 30 | Recorded |
| **Failed/Rate-Limited Requests** | 0 | N/A | Recorded |

## 3. Dataset Collection Manifest
- **Queries Tested**: 15 queries across 7 distinct categories.
- **Models Evaluated**: 3 models (1 local Ollama baseline + 2 fixed OpenRouter models).
- **Total Raw Output Records**: 45 complete un-truncated records stored in `evaluation/results/phase5b2_raw_model_outputs.jsonl`.
- **Request Ledger Stored**: `evaluation/results/phase5b2_request_ledger.json`.
- **Frozen Context Manifest Stored**: `evaluation/results/phase5b2_frozen_context_manifest.json`.
- **Local Judge Results Stored**: `evaluation/results/phase5b2_raw_judge_results.jsonl` (45 local Ollama judge outputs).

## 4. Final Status Block
```text
PHASE_5B2B_STATUS: COMPLETE
OPENROUTER_HARD_LIMIT: 40
OPENROUTER_REQUESTS_USED: 30
OPENROUTER_REQUESTS_REMAINING: 10
OPENROUTER_REQUEST_LIMIT_RESPECTED: PASS
OPENROUTER_MODELS_TESTED: 2
OLLAMA_BASELINE_COLLECTED: PASS
QUERIES_TESTED: 15
FROZEN_CONTEXT_CONTROL: PASS
PROMPT_HASH_CONTROL: PASS
CONTEXT_HASH_CONTROL: PASS
RAW_ANSWERS_STORED: PASS
TOKEN_USAGE_STORED: PASS
LATENCY_STORED: PASS
MODEL_IDENTITY_STORED: PASS
REQUEST_IDS_STORED: PASS
JUDGE_RAW_OUTPUT_STORED: PASS
DOCUMENT_SCOPE_EVIDENCE: PASS
REFUSAL_EVIDENCE: PASS
LANGSMITH_EVIDENCE: PASS
LOCAL_TRACE_EVIDENCE: PASS
PRODUCTION_FILES_MODIFIED: 0
PRODUCTION_ROUTING_CHANGED: NO
MODEL_SELECTION_DECISION: DEFERRED
DATA_READY_FOR_PHASE_5B2C_ANALYSIS: YES
```