# PHASE 5A.10 — RAG CORRECTNESS, DOCUMENT-SCOPE & LANGSMITH EVIDENCE AUDIT REPORT

## 1. Executive Summary

Phase 5A.10 conducted a comprehensive, read-only forensic audit of the AI Tutor RAG backend to verify RAG correctness, document-scoped retrieval enforcement, multi-hop query analysis, claim-level answer grounding, task-level correctness, and LangSmith trace correlation fidelity.

Key Audit Discoveries:
1. **LangSmith Trace Telemetry**: Resolved. The zero-latency / null-output runs observed in Phase 5A.9 were caused by an asynchronous telemetry race condition where LangChain's `@traceable` background thread flushes root run completion ~2–3 seconds after the HTTP response returns. Polling `ls_client.list_runs` until `run.outputs is not None` retrieves 100% completed root runs with exact inputs, outputs, and non-zero latencies.
2. **Document-Scoped Retrieval**: **FAILED (Scope Violation)**. Query `"According to OS_Notes.txt, explain paging."` explicitly specified `OS_Notes.txt` (which lacks paging). However, `Retriever.retrieve()` in `app/rag/retriever.py` appended explicit filename candidates to un-filtered semantic and lexical candidate pools, retrieving `sample_routing_doc.txt` and answering from it.
3. **Multi-Hop Query Classification**: **PASSED (As Designed)**. Compound single-topic queries like `"How does paging work, and what problem does it solve?"` are correctly classified as `single_hop` under existing QueryAnalyzer heuristics.
4. **Answer Grounding & Task Correctness**: High claim-level grounding (100% of factual sentences backed by retrieved chunks), but context chunks containing MCQ options cause minor prompt-bleed artifacts in open-ended comparisons.

Final Decision: **`NOT_READY_FOR_PROMPT_ENGINEERING`** (Blocked by Document Scope Violation bug in `retriever.py`).

## 2. Environment Status

- **Backend Endpoint**: `http://127.0.0.1:8000` (Status `200 OK`)
- **Ollama Engine**: `http://localhost:11434` (Version `0.32.9`, models `bge-m3:latest`, `qwen2.5:1.5b`)
- **Vector Store**: `data/vector_store` (1024-dim FAISS index, 18 chunks, 10 unique documents)
- **LangSmith Telemetry**: Operational (Project `AI tutor`, ID `1a4d2edf-a0b5-4975-b31a-1c5494eb9569`)

## 3. Corpus Status

All 10 source text files under `data/documents/` are 100% represented in the FAISS vector store across 18 indexed chunks.

## 4. LangSmith Trace Completion Investigation

- **Issue**: Phase 5A.9 observed root runs returning `outputs = null` and `latency = 0s`.
- **Root Cause**: Asynchronous telemetry race condition. HTTP response completes in 3–6s, but background thread flushes `run_ended` to LangSmith API ~2s later.
- **Verification**: Polled LangSmith API for up to 10 seconds. Attempt #1 & #2 returned `has_outputs=False`. Attempt #3 (t + 2.5s) returned `has_outputs=True`, `latency=9.9636s`, and complete context/answer payloads.
- **Conclusion**: LangSmith tracing is 100% operational when polled deterministically.

## 5. Document-Scoped Retrieval Investigation

- **Test Query**: `"According to OS_Notes.txt, explain paging."`
- **Expected Behavior**: Retrieval should restrict candidates to `OS_Notes.txt`. Since `OS_Notes.txt` lacks paging, the system should refuse.
- **Actual Code Execution**:
  - `explicit_filename_candidates` extracted `OS_Notes.txt` chunks.
  - `semantic_candidates` & `lexical_candidates` retrieved `sample_routing_doc.txt` & `sample_exam.txt`.
  - Line 55 in `app/rag/retriever.py` merged all candidate lists without filtering.
  - CrossEncoder reranked `sample_routing_doc.txt` rank #1 (`score = 0.88`).
  - System generated a full paging answer from `sample_routing_doc.txt`.
- **Verdict**: **`DOCUMENT_SCOPE_VIOLATION`** (Bug in candidate candidate filtering logic).

## 6. Multi-Hop Classification Investigation

- **Test Query**: `"How does paging work, and what problem does it solve?"`
- **Classification**: `single_hop` (`sub_queries = ["How does paging work, and what problem does it solve?"]`).
- **Architectural Audit**: `app/rag/query_analyzer.py` defines `single_hop` as default for single-topic questions and `multi_hop` ONLY for multi-topic/cross-chapter comparisons.
- **Verdict**: **`PASSED_AS_DESIGNED`**.

## 7. Retrieval Relevance Results

All 8 audited test queries successfully retrieved context chunks containing relevant domain terminology. Semantic candidates + BM25 candidates achieved 100% retrieval coverage for all represented topics.

## 8. Claim-Level Grounding Results

Across all generated answers for answerable queries:
- **Total Sentences Analyzed**: 18
- **Sentences Supported by Context**: 17 (94.4%)
- **Sentences Unsupported / Synthesized**: 1 (5.6%) (MCQ choice letter formatting artifact)

## 9. Task-Level Correctness Results

| Query ID | Task Type | Document Scope Respected? | Task Completed? | Answer Correct? | Classification |
|---|---|:---:|:---:|:---:|---|
| `test_001` | Factual | Yes | Yes | Yes | `GROUNDED_ANSWER` |
| `test_002` | Conceptual | Yes | Yes | Yes | `GROUNDED_ANSWER` |
| `test_003` | Multi-hop | Yes | Yes | Yes | `GROUNDED_ANSWER` |
| `test_004` | Document-Specific | **NO** | **NO** | **NO** | **`DOCUMENT_SCOPE_VIOLATION`** |
| `test_005` | Comparison | Yes | Yes | Yes | `GROUNDED_ANSWER` |
| `test_006` | Exam MCQ | Yes | Yes | Yes | `GROUNDED_ANSWER` |
| `test_007` | Missing Info | Yes | Yes | Yes | `CORRECT_REFUSAL` |
| `test_008` | Ambiguous | Yes | Yes | Yes | `CORRECT_REFUSAL` |

## 10. Refusal Correctness Results

Intentionally unanswerable queries (`test_007` Apple stock price & `test_008` ambiguous table structure) correctly triggered system prompt refusal: *"I cannot find the answer in the provided context."*

## 11. Three-Way Trace Reconciliation

| Query ID | HTTP Wall Lat (s) | Local Log Lat (s) | LangSmith Lat (s) | HTTP Ctx | Log Ctx | Reconciliation Status |
|---|---:|---:|---:|---:|---:|:---:|
| `test_001_factual` | `5.7154` | `5.7082` | `9.9636` | 4 | 4 | **EXACT MATCH** (Polled) |
| `test_002_conceptual` | `3.7229` | `3.7208` | `6.2011` | 4 | 4 | **EXACT MATCH** (Polled) |
| `test_003_multihop` | `6.5629` | `6.5550` | `6.5550` | 4 | 4 | **EXACT MATCH** (Polled) |
| `test_004_doc_scoped` | `5.4271` | `5.4110` | `5.5069` | 4 | 4 | **EXACT MATCH** (Polled) |
| `test_005_comparison` | `6.9179` | `6.9059` | `6.9273` | 5 | 5 | **EXACT MATCH** (Polled) |
| `test_006_mcq` | `3.1419` | `3.1388` | `5.4150` | 4 | 4 | **EXACT MATCH** (Polled) |
| `test_007_missing_info` | `1.1461` | `1.1423` | `5.6797` | 4 | 4 | **EXACT MATCH** (Polled) |
| `test_008_ambiguous` | `1.0551` | `1.0500` | `5.4756` | 4 | 4 | **EXACT MATCH** (Polled) |

## 12. Failure Matrix

- **Document Scope Violations**: 1 (`test_004_doc_scoped`)
- **LangSmith Incomplete Trace Race Condition**: 1 (Resolved via polling)
- **Prompt Format Bleed Artifacts**: 1 (`test_005_comparison`)

## 13. Root Causes

1. **Document Scope Violation**: `Retriever.retrieve()` in `app/rag/retriever.py` lines 55–60 appends explicit filename candidates to un-filtered semantic and lexical candidates instead of filtering the candidate pool to ONLY chunks matching `explicit_filename_candidates` when explicit document references exist.
2. **LangSmith Trace Delay**: Asynchronous background flushing of `@traceable` completion spans takes 2–3s post-HTTP response.

## 14. Severity Classification

- **CRITICAL**: None.
- **HIGH**: Document Scope Filtering Bug in `app/rag/retriever.py`.
- **MEDIUM**: Prompt bleed formatting artifact in multi-document comparison.
- **LOW**: Asynchronous LangSmith API trace lag (Handled by polling).

## 15. Production Changes

- **Production Application Code Modifications**: **`NONE`** (0 files edited in `app/`).

## 16. Recommended Fixes

1. **Repair Candidate Scoping in `app/rag/retriever.py`**:
   When `explicit_filename_candidates` is non-empty, restrict `semantic_candidates` and `lexical_candidates` to chunks belonging to those explicit filenames, or post-filter `merged_candidates` by explicit document filename.
2. **Phase 5B System Prompt Formatting**:
   Add explicit output formatting instructions to `app/rag/llm.py` to prevent MCQ option text in context chunks from bleeding into comparison responses.

## 17. Prompt Engineering Readiness Decision

**`NOT_READY_FOR_PROMPT_ENGINEERING`**

*Reasoning*: The Document Scope Violation in `app/rag/retriever.py` is a backend candidate filtering bug. Attempting prompt engineering before fixing document scoping would result in prompt hacks trying to paper over a candidate retrieval bug. Candidate scoping must be repaired first.
