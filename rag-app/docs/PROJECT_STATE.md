# AI Tutor RAG — Authoritative Project State

| Attribute | Value |
|---|---|
| **Last Verified Date** | `2026-08-13` |
| **Current Phase** | `Phase 5B.2E (Completed)` |
| **Current Status** | `VERIFIED & FROZEN` |
| **Current Production Routing** | `ALL_OLLAMA` (`qwen2.5:1.5b`) |
| **Document Version** | `1.0.0` |
| **Evidence Confidence** | `HIGH` |

---

## 1. Project Overview

### Problem Statement
The **AI Tutor RAG Application** is a specialized Retrieval-Augmented Generation system designed for processing educational courseware, textbook chapters, exam review notes, and technical computer science documentation. Its goal is to provide accurate, grounded answers, explain complex concepts, synthesize cross-document information, and maintain zero-hallucination boundaries when answering student queries.

### Intended User
Students, educators, and automated tutoring systems seeking grounded QA over technical documentation without hallucinated answers or out-of-scope document leaks.

### Current RAG Capabilities
- **Two-Stage Hybrid Retrieval**: Combines BGE-M3 vector semantic search with BM25 lexical search, merged via content deduplication.
- **CrossEncoder Reranking**: Uses `BAAI/bge-reranker-v2-m3` to re-score candidate chunks before context assembly.
- **Query Analysis & Decomposition**: Classifies incoming queries as `single_hop` or `multi_hop` and decomposes multi-topic queries into sub-queries.
- **Strict Document-Scope Enforcement**: Restricts candidate retrieval exclusively to requested document filenames when explicit document references exist (`FINAL_CONTEXT.documents ⊆ REQUESTED_DOCUMENTS`). Unscoped queries continue to search globally across all indexed documents.
- **Context-Boundary Refusal**: Refuses unanswerable or out-of-domain queries with a standardized refusal message (*"I cannot find the answer in the provided context."*).
- **Comprehensive Observability**: Full three-way trace reconciliation across FastAPI HTTP endpoints, structured local log files (`logs/rag_traces.log`), and LangSmith platform tracing.

### Scope & Invariants
- **Corpus Scope**: 10 computer science reference documents (18 chunks total indexed in FAISS).
- **Production Scope**: Local generation via Ollama `qwen2.5:1.5b`. External LLM providers (e.g. OpenRouter) have been integrated and evaluated but rejected for production routing based on empirical quality/latency benchmarks.

### Implemented vs. Non-Implemented
- **Implemented**: FastAPI backend, LangGraph workflow, FAISS vector store, BM25 retriever, CrossEncoder reranker, document-scope filtering, OpenRouter provider integration, local judge evaluation framework, LangSmith observability telemetry.
- **Not Implemented**: Production OpenRouter routing (rejected at Phase 5B.2E gate), paid commercial API model routing, streaming response UI integration, multi-user concurrent session management.

---

## 2. Current Project Status

| Area | Status | Evidence |
|---|---|---|
| **Backend API** | `COMPLETE` / `VERIFIED` | FastAPI application (`app/main.py`), 100% HTTP 200 OK across baseline runs |
| **RAG Orchestration** | `COMPLETE` / `VERIFIED` | LangGraph workflow (`app/graph/workflow.py`, `app/graph/nodes.py`) |
| **Retrieval System** | `COMPLETE` / `VERIFIED` | Hybrid FAISS (bge-m3) + BM25 + CrossEncoder (`app/rag/retriever.py`) |
| **Vector Store Index** | `COMPLETE` / `VERIFIED` | 100% consistency across 10 disk documents / 18 FAISS chunks (`data/vector_store`) |
| **Document Scoping** | `COMPLETE` / `VERIFIED` | Phase 5A.11 candidate filtering fix verified by 6/6 regression tests |
| **LLM Service** | `COMPLETE` / `VERIFIED` | Ollama `qwen2.5:1.5b` via ChatOllama (`app/rag/llm.py`) |
| **OpenRouter Provider** | `VERIFIED` / `REJECTED FOR PROD` | Provider integrated (`app/rag/openrouter_provider.py`); rejected at 5B.2E gate |
| **Evaluation Framework** | `COMPLETE` / `VERIFIED` | Local judge suite (`scratch/phase5b2b_local_judge_eval.py`), 45 judge outputs |
| **LangSmith Tracing** | `COMPLETE` / `VERIFIED` | Tracing enabled (`project: tutor-rag-backend`), 100% trace completion via polling |
| **Prompt Engineering** | `FROZEN` | Standardized grounding prompt unchanged throughout Phase 5 evaluation |
| **Frontend** | `COMPLETE` | Streamlit UI (`rag-app/frontend/app.py`) for query and document management |

---

## 3. Architecture

```text
               User Query
                   │
                   ▼
             FastAPI Endpoint
           (POST /query in app/main.py)
                   │
                   ▼
           LangGraph Workflow
            (RAGGraph in app/graph/workflow.py)
                   │
                   ▼
            Query Analyzer
        (QueryAnalyzer in app/rag/query_analyzer.py)
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  [single_hop]           [multi_hop]
(Single Query)         (Sub-Queries)
        │                     │
        └──────────┬──────────┘
                   ▼
            Hybrid Retriever
         (Retriever in app/rag/retriever.py)
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
Explicit      BGE-M3 Vector    BM25 Lexical
Filename        Semantic         Search
Matching         Search        (Top 15)
 (Doc Check)    (Top 15)          │
    │              │              │
    └──────────────┴──────────────┘
                   │
                   ▼
     Document Scope Filter (Phase 5A.11)
  (If filename requested: restrict candidates 
       strictly to requested document)
                   │
                   ▼
        Deduplication & Merging
                   │
                   ▼
    CrossEncoder Reranker (top_k=4)
  (BAAI/bge-reranker-v2-m3 in app/rag/reranker.py)
                   │
                   ▼
             Final Context
                   │
                   ▼
             LLM Generation
      (LLMService in app/rag/llm.py)
      (Ollama qwen2.5:1.5b @ temp=0.1)
                   │
                   ▼
           Generated Answer
                   │
    ┌──────────────┴──────────────┐
    ▼                             ▼
Local Log Trace              LangSmith API
(logs/rag_traces.log)    (Project: tutor-rag-backend)
```

### Component Responsibilities & Boundaries
1. **`app/main.py`**: Application bootstrap, logging configuration, instance initialization (`EmbeddingService`, `VectorStore`, `BM25Retriever`, `Reranker`, `Retriever`, `QueryAnalyzer`, `LLMService`, `RAGNodes`, `RAGGraph`).
2. **`app/config.py`**: Configuration loading via Pydantic `BaseSettings` reading environment parameters and secrets.
3. **`app/rag/query_analyzer.py`**: Rule/LLM heuristic classification into `single_hop` or `multi_hop` with sub-query generation.
4. **`app/rag/retriever.py`**: Orchestrates candidate collection (filename match + FAISS vector + BM25 lexical), applies candidate document-scope filtering, deduplicates, and passes candidates to CrossEncoder.
5. **`app/rag/llm.py`**: Prepares context-grounded prompt, invokes active provider (`LLMProvider`), records generation metadata, and attaches LangSmith span attributes.
6. **`app/rag/openrouter_provider.py`**: Custom HTTP client communicating with OpenRouter `/chat/completions` API with header sanitization and ledger recording.

---

## 4. Repository Structure

```text
rag-app/
├── README.md
├── frontend/
│   ├── app.py                      # Streamlit frontend UI
│   └── requirements.txt
└── backend/
    ├── app/                        # Application source code
    │   ├── main.py                 # FastAPI entry point & DI container
    │   ├── config.py               # Pydantic Settings configuration
    │   ├── api/
    │   │   └── routes.py           # FastAPI routes (/health, /query, /documents/upload)
    │   ├── models/
    │   │   ├── canonical.py        # Document & Chunk models
    │   │   └── state.py            # LangGraph state definitions (RAGState, QueryAnalysis)
    │   ├── graph/
    │   │   ├── nodes.py            # LangGraph step functions (analyze, route, retrieve, generate)
    │   │   └── workflow.py         # StateGraph build & compilation
    │   └── rag/
    │       ├── bm25_retriever.py   # BM25 lexical search implementation
    │       ├── document_loader.py  # Text document loading
    │       ├── document_splitter.py# Chunking logic (chunk_size=500, overlap=50)
    │       ├── embeddings.py       # Ollama BGE-M3 embedding wrapper & query cache
    │       ├── llm.py              # Core LLM generation service
    │       ├── openrouter_provider.py # Standalone OpenRouter API integration
    │       ├── providers.py        # Abstract LLMProvider interface & registry
    │       ├── query_analyzer.py   # Single-hop vs multi-hop query classifier
    │       ├── reranker.py         # CrossEncoder (bge-reranker-v2-m3) reranker
    │       ├── retriever.py        # Hybrid retriever & document scope filter
    │       └── vector_store.py     # FAISS vector store wrapper
    ├── data/
    │   ├── documents/              # 10 source text files
    │   └── vector_store/           # FAISS index.faiss & index.pkl (18 chunks)
    ├── evaluation/
    │   ├── datasets/               # Evaluation datasets (rag_baseline_v1.jsonl)
    │   └── results/                # Phase 5A and Phase 5B audit artifacts & JSON ledgers
    ├── logs/
    │   └── rag_traces.log          # Structured local execution logs
    ├── scratch/                    # Audit runners, benchmark scripts, and verification code
    ├── performance_phase5a*.md     # Historical Phase 5A markdown audit reports
    └── requirements.txt
```

---

## 5. RAG Pipeline Details

### Query Analysis
- **`single_hop`**: Default classification for single-concept or direct queries. Returns `sub_queries = [original_query]`.
- **`multi_hop`**: Triggered when query contains explicit cues (`compare`, `difference between`, `versus`, `vs`, `chapter X ... chapter Y`). Generates 2–3 sub-queries.
- **Routing**: `retrieve_single` runs 1 retrieval pass; `retrieve_multi` executes retrieval per sub-query and merges unique context chunks (capped at 6).

### Retrieval Architecture
1. **Candidate Retrieval**:
   - **Semantic Search**: FAISS vector search using `nomic-embed-text` / `bge-m3` embeddings (`top_k_candidates = 15`).
   - **Lexical Search**: BM25 search over docstore (`top_k_candidates = 15`).
   - **Explicit Filename Matching**: Direct substring check against indexed `source_filename` metadata.
2. **Document-Scope Enforcement (Phase 5A.11 Invariant)**:
   - When explicit filename candidates exist, `allowed_filenames` set is built.
   - `semantic_candidates` and `lexical_candidates` are filtered to ONLY include chunks matching `allowed_filenames`.
   - **Invariant Guarantee**: `FINAL_CONTEXT.documents ⊆ REQUESTED_DOCUMENTS`. Unscoped queries maintain `allowed_filenames = set()`, searching globally across all documents.
3. **Merging & Deduplication**: Candidate chunks are merged and deduplicated using exact chunk text strings.
4. **CrossEncoder Reranking**: Candidate chunks are passed to `BAAI/bge-reranker-v2-m3`, sorted by score, and cut off at `top_k_final = 4`.

### Generation & Grounding
- **Grounding Prompt**:
  ```text
  Answer the question based ONLY on the supplied context.
  If the answer is not supported by the context, state clearly:
  "I cannot find the answer in the provided context."
  ```
- **Refusal Behavior**: When top candidate reranker scores are low or context lacks requested facts, model generates standardized refusal text.

---

## 6. Current Models

| Role | Provider | Model Identifier | Status | Operational Notes |
|---|---|---|---|---|
| **Answer Generator** | `ollama` | `qwen2.5:1.5b` | **`PRODUCTION`** | 100% production traffic; 4.07 quality score, 3.00s latency |
| **Query Analyzer** | `ollama` | `qwen2.5:1.5b` | **`PRODUCTION`** | Runs single_hop / multi_hop analysis |
| **Text Embeddings** | `ollama` | `bge-m3` (1024-dim) | **`PRODUCTION`** | Used for FAISS vector index & query embeddings |
| **Reranker** | `sentence-transformers` | `BAAI/bge-reranker-v2-m3` | **`PRODUCTION`** | Local CrossEncoder model for candidate reranking |
| **Candidate Model A** | `openrouter` | `nvidia/nemotron-3.5-lightning:free` | **`REJECTED`** | Evaluated in Phase 5B.2; rejected due to high latency (39.73s) |
| **Candidate Model B** | `openrouter` | `nvidia/nemotron-nano-9b-v2:free` | **`REJECTED`** | Evaluated in Phase 5B.2; rejected due to quality/latency penalty |

---

## 7. OpenRouter Configuration

- **Configuration Properties**:
  - `OPENROUTER_API_KEY`: `[CONFIGURED — SECRET NOT DOCUMENTED]`
  - `OPENROUTER_MODEL`: `"openrouter/free"` (or specific model identifiers during benchmarks)
  - `OPENROUTER_BASE_URL`: `"https://openrouter.ai/api/v1"`
- **Loading & Security**: Loaded via `app/config.py` Pydantic `Settings`. Provider implementation (`app/rag/openrouter_provider.py`) strips Authorization headers from exception tracebacks and return objects.
- **Production Status**: Configured and validated via `OpenRouterProvider`, but **NOT enabled for production routing**. Phase 5B.2E gate established `ROUTING_DECISION: ALL_OLLAMA`.

---

## 8. Evaluation Methodology

To evaluate model quality fairly without retrieval noise, Phase 5B.2 used a **Same-Context Frozen Retrieval A/B Benchmark**:

1. **Frozen Retrieval**: RAG retrieval was executed ONCE per query. Context chunks were serialized and frozen into `phase5b2_frozen_context_manifest.json`.
2. **Experimental Controls**:
   - `SHA256(prompt)` and `SHA256(final_context)` were verified to be 100% identical across all model runs per query.
   - Raw model outputs were stored un-truncated in `phase5b2_raw_model_outputs.jsonl`.
3. **Local Judge Evaluation**:
   - Evaluated using a local Ollama judge (`qwen2.5:1.5b`) running deterministically (`temperature=0.0`) using structured JSON evaluation prompts.
   - **Zero paid judge API calls** were made.
   - Outputs saved in `phase5b2_raw_judge_results.jsonl`.
4. **Scored Dimensions (0.0 to 5.0)**: Correctness, Grounding, Completeness, Relevance, Instruction Following, Overall Quality (mean of 5 dimension scores).
5. **Observability**: Recorded latency, token counts, request IDs, and three-way trace correlation (HTTP ↔ local log ↔ LangSmith).

---

## 9. Benchmark Dataset

The Phase 5B.2B dataset comprises **15 representative queries** across 8 distinct categories, generating **45 raw model output records** and **45 judge evaluation records**:

### Category Breakdown
- `factual`: 3 queries (`factual_001`, `factual_002`, `factual_003`)
- `conceptual`: 3 queries (`concept_001`, `concept_002`, `concept_003`)
- `multi_hop`: 3 queries (`multihop_001`, `multihop_002`, `multihop_004`)
- `document_specific`: 2 queries (`doc_001`, `doc_003`)
- `comparison`: 1 query (`compare_001`)
- `missing_information`: 1 query (`missing_002`)
- `multi_document`: 1 query (`multidoc_001`)
- `exam_style`: 1 query (`exam_001`)

### Budget & Ledger Summary
- **Hard API Request Limit**: 40 requests
- **OpenRouter Requests Used**: 30 requests (15 for Model A + 15 for Model B)
- **OpenRouter Budget Remaining**: 10 requests remaining at validation close

---

## 10. FINAL MODEL QUALITY METRICS

*Reconciled Phase 5B.2E Final Benchmark Results (15 Benchmark Queries, 45 Total Evaluation Records):*

| Metric | Local Ollama (`qwen2.5:1.5b`) | OpenRouter Model A (`Nemotron 3.5 Lightning`) | OpenRouter Model B (`Nemotron Nano 9B`) |
|---|:---:|:---:|:---:|
| **Correctness (0-5)** | **3.93** | 3.27 | 3.47 |
| **Grounding (0-5)** | **4.20** | 2.73 | 3.27 |
| **Completeness (0-5)** | 3.80 | **3.93** | 3.87 |
| **Relevance (0-5)** | 3.67 | **3.73** | **3.73** |
| **Instruction Following (0-5)** | **4.73** | 4.27 | 4.47 |
| **Overall Quality (Mean)** | **4.07** | 3.59 | 3.76 |
| **Overall Quality (Median)** | **4.40** | 4.00 | 4.00 |
| **Mean Latency (sec)** | **3.00s** | 39.73s | 12.64s |
| **Median Latency (sec)** | **2.84s** | 29.36s | 7.67s |
| **P95 Latency (sec)** | **4.06s** | 95.16s | 41.54s |
| **Latency Multiplier vs Ollama** | **1.0x** | 13.2x | 4.2x |
| **Mean Total Tokens** | **381.20** | 2011.67 | 689.87 |
| **Token Multiplier vs Ollama** | **1.0x** | 5.3x | 1.8x |

---

## 11. Pairwise Head-to-Head Results

### 1. Local Ollama vs OpenRouter Model A (Nemotron 3.5 Lightning)
- **Ollama Wins**: **9** (60.0%)
- **Model A Wins**: **3** (20.0%)
- **Ties**: **3** (20.0%)
- **Mean Overall Quality Delta**: **+0.48** in favor of Ollama

### 2. Local Ollama vs OpenRouter Model B (Nemotron Nano 9B)
- **Ollama Wins**: **5** (33.3%)
- **Model B Wins**: **3** (20.0%)
- **Ties**: **7** (46.7%)
- **Mean Overall Quality Delta**: **+0.31** in favor of Ollama

### 3. OpenRouter Model A vs OpenRouter Model B
- **Model A Wins**: **3** (20.0%)
- **Model B Wins**: **7** (46.7%)
- **Ties**: **5** (33.3%)
- **Mean Overall Quality Delta**: **+0.17** in favor of Model B

### Interpretation
Local Ollama outperforms both OpenRouter candidates head-to-head in quality while operating 4.2x to 13.2x faster.

---

## 12. Category-Level Findings

| Category | Query Count | Ollama Quality | Model A Quality | Model B Quality | Category Winner | Evidence Confidence |
|---|:---:|:---:|:---:|:---:|---|---|
| `comparison` | 1 | **4.40** | 4.00 | 3.80 | **`OLLAMA_BASELINE`** | `INSUFFICIENT_EVIDENCE` (N=1) |
| `conceptual` | 3 | **4.13** | 3.20 | **4.13** | **`TIE`** | `NO_CLEAR_WINNER` |
| `document_specific` | 2 | 3.30 | 3.50 | **4.00** | **`OPENROUTER_MODEL_B`** | `MEDIUM` |
| `exam_style` | 1 | 4.00 | 2.40 | **4.40** | **`OPENROUTER_MODEL_B`** | `INSUFFICIENT_EVIDENCE` (N=1) |
| `factual` | 3 | 3.93 | **4.27** | 3.40 | **`OPENROUTER_MODEL_A`** | `MEDIUM` |
| `missing_information` | 1 | **4.60** | 1.60 | 3.00 | **`OLLAMA_BASELINE`** | `INSUFFICIENT_EVIDENCE` (N=1) |
| `multi_document` | 1 | 4.00 | **4.40** | **4.40** | **`TIE`** | `INSUFFICIENT_EVIDENCE` (N=1) |
| `multi_hop` | 3 | **4.40** | 4.00 | 3.40 | **`OLLAMA_BASELINE`** | `MEDIUM` |

### Key Tradeoff Observations
- **`document_specific`**: Model B scored +0.70 higher overall quality, but required 12.64s latency (4.2x latency penalty).
- **`factual`**: Model A scored +0.33 higher overall quality, but required 39.73s latency (13.2x latency penalty).
- **Verdict**: In no category did an OpenRouter model deliver a large enough quality advantage to outweigh the 4x to 13x latency slowdown and external dependency risks.

---

## 13. Refusal Analysis & Reconciliation

Historical reports contained different refusal metrics due to different query subsets and definitions. Phase 5B.2E explicitly reconciled these into 3 distinct definitions:

### Definition 1: All-Query Refusal Accuracy (All 15 Benchmark Queries)
Evaluates whether refusal behavior was correct across both answerable and unanswerable queries.
- **Local Ollama**: 12 / 15 correct (**80.0%**)
- **OpenRouter Model A**: 4 / 15 correct (**26.7%**)
- **OpenRouter Model B**: 8 / 15 correct (**53.3%**)

### Definition 2: Explicit Refusal Target Queries (`missing_002` & `doc_001`)
Evaluates accuracy on the 2 benchmark queries targeting missing information.
- **Local Ollama**: 1 / 2 correct (**50.0%**)
- **OpenRouter Model A**: 1 / 2 correct (**50.0%**)
- **OpenRouter Model B**: 1 / 2 correct (**50.0%**)
- *Note on Discrepancy*: `doc_001` expected `scoped_retrieval_answer` in the benchmark spec, but `OS_Notes.txt` lacked paging text, leading all models to refuse correctly according to context boundaries, but counting as 50% against the legacy spec.

### Definition 3: Single Out-Of-Domain Refusal Query (`missing_002` Apple Stock Price)
Evaluates refusal accuracy on pure out-of-domain queries.
- **Local Ollama**: 1 / 1 correct (**100.0%**)
- **OpenRouter Model A**: 1 / 1 correct (**100.0%**)
- **OpenRouter Model B**: 1 / 1 correct (**100.0%**)

---

## 14. Retrieval & Document-Scope Validation

Following the Phase 5A.10 bug discovery and Phase 5A.11 candidate filtering fix in `app/rag/retriever.py`, the document-scope regression suite was re-executed:

| Test ID | Query | Requested Scope | Final Context Documents | Scope Invariant | Status |
|---|---|---|---|:---:|:---:|
| `test_1` | `"What is virtual memory?"` | `ALL_DOCUMENTS` | `OS_Notes.txt`, `sample_exam.txt`, `sample_exam_inspect.txt` | `True` | **`PASS`** |
| `test_2` | `"According to OS_Notes.txt, explain paging."` | `OS_Notes.txt` | `OS_Notes.txt` | `True` | **`PASS`** |
| `test_3` | `"According to OS_Notes.txt, what is virtual memory?"` | `OS_Notes.txt` | `OS_Notes.txt` | `True` | **`PASS`** |
| `test_4` | `"According to sample_routing_doc.txt, explain paging."` | `sample_routing_doc.txt` | `sample_routing_doc.txt` | `True` | **`PASS`** |
| `test_5` | `"Compare memory management in OS_Notes.txt and sample_exam.txt."` | `OS_Notes.txt`, `sample_exam.txt` | `OS_Notes.txt`, `sample_exam.txt` | `True` | **`PASS`** |
| `test_6` | `"Explain paging."` | `ALL_DOCUMENTS` | `sample_routing_doc.txt`, `sample_exam.txt` | `True` | **`PASS`** |

**Invariant Compliance**: 100% compliance with `FINAL_CONTEXT.documents ⊆ REQUESTED_DOCUMENTS`. Zero non-requested document leaks occurred.

---

## 15. Observability & Evidence

### 1. HTTP Endpoint Telemetry
FastAPI `/query` endpoint returns structured response payloads containing `answer`, `context`, `query_type`, `sub_queries`, and execution metadata.

### 2. Local Trace Logging
`logs/rag_traces.log` records structured `TRACE SUMMARY` lines containing timestamp, question, query type, retrieval candidate counts, reranker scores, latency, and context metadata.

### 3. LangSmith Tracing
- **Project**: `tutor-rag-backend` (ID: `1a4d2edf-a0b5-4975-b31a-1c5494eb9569`)
- **Race Condition Resolution**: Resolved the Phase 5A.10 telemetry race condition where `@traceable` background threads flushed spans 2–3s post-HTTP response. Deterministic polling (`ls_client.list_runs` until `run.outputs is not None`) achieves 100% trace retrieval.

---

## 16. Phase-by-Phase History

### Phase 5A.7 — Corpus & Index Audit
- **Objective**: Audit source documents, FAISS index, and 35 baseline queries.
- **Findings**: Found 9 of 10 disk documents unindexed in FAISS. 17 queries lacked text in source corpus.
- **Classification**: `CORPUS_INCOMPLETE`. Zero production edits made.

### Phase 5A.8 — Vector Store Rebuild
- **Objective**: Rebuild FAISS index with all 10 disk files via `POST /documents/upload`.
- **Result**: FAISS expanded from 1 chunk to 18 chunks. Grounded answers increased from 11 to 21 (+90.9%). Backup created in `data/vector_store_backup_phase5a8_*`.

### Phase 5A.9 — Final Corpus Validation
- **Objective**: Verify 100% reconciliation of 10 disk files and 18 FAISS chunks.
- **Result**: Three-way trace consistency (HTTP, log, LangSmith) verified across 35 queries.

### Phase 5A.10 — RAG Correctness & Scoping Audit
- **Objective**: Audit document scoping, multi-hop routing, and claim grounding.
- **Findings**: Discovered `DOCUMENT_SCOPE_VIOLATION` bug in `retriever.py` (candidate candidates merged without filtering). Discovered LangSmith async telemetry lag.
- **Classification**: `NOT_READY_FOR_PROMPT_ENGINEERING`.

### Phase 5A.11 — Document Scope Production Fix
- **Objective**: Fix candidate document scoping bug in `app/rag/retriever.py`.
- **Changes**: Added 12 lines of candidate document filtering when `explicit_filename_candidates` exist.
- **Result**: Passed 6/6 regression tests (`FINAL_CONTEXT.documents ⊆ REQUESTED_DOCUMENTS`).

### Phase 5B.1A — Hybrid Routing Audit & OpenRouter Integration
- **Objective**: Implement dynamic complexity-based hybrid LLM routing and audit OpenRouter connectivity.
- **Changes**: Added `OpenRouterProvider` (`app/rag/openrouter_provider.py`), role-based settings, and dynamic routing in `RAGNodes`. Passed 35/35 three-way reconciliation.

### Phase 5B.2A — Same-Context Model A/B Benchmark
- **Objective**: Conduct initial same-context benchmark across 16 queries.
- **Result**: Prompt and context SHA256 hashes verified 100% identical across model runs.

### Phase 5B.2B — Controlled Dataset Collection
- **Objective**: Collect un-truncated raw evaluation dataset under a 40-request API budget cap.
- **Result**: Used 30 OpenRouter requests (10 remaining). Generated 45 model outputs and 45 local judge outputs across 15 queries.

### Phase 5B.2C — Schema & Model Quality Analysis
- **Objective**: Analyze frozen Phase 5B.2B dataset with parser schema corrections for top-level judge fields.
- **Result**: Established base quality scores (Ollama 4.07, Model B 3.76, Model A 3.59).

### Phase 5B.2D — Deep Category & Routing Analysis
- **Objective**: Analyze model performance across 8 query categories and evaluate latency tradeoffs.
- **Result**: Demonstrated that routing to OpenRouter free models increases latency by 4x to 13x without improving overall quality.

### Phase 5B.2E — Final Reconciliation & Gate Report
- **Objective**: Reconcile all refusal metric definitions, execute final routing gate evaluation, and establish production policy.
- **Result**: Closed all contradictions. Issued final decision: `ROUTING_DECISION: ALL_OLLAMA`.

---

## 17. Production Changes Audit

| Phase | Production Files Modified | Change Details |
|---|---|---|
| Phase 5A.7 | `NONE` | Read-only audit |
| Phase 5A.8 | `NONE` | Vector index rebuild executed via HTTP upload API |
| Phase 5A.9 | `NONE` | Read-only validation |
| Phase 5A.10 | `NONE` | Read-only audit |
| Phase 5A.11 | `app/rag/retriever.py` | Added candidate document filtering when explicit filename candidates exist (12 lines) |
| Phase 5B.1A | `app/config.py`, `app/main.py`, `app/graph/nodes.py`, `app/rag/llm.py`, `app/rag/openrouter_provider.py`, `app/rag/providers.py` | Integrated OpenRouter provider & role-based LLM routing |
| Phase 5B.2A-E | `NONE` | All evaluation phases executed offline on frozen datasets with zero production code edits |

---

## 18. Current Routing Decision

```text
============================================================
FINAL PRODUCTION ROUTING GATE (Phase 5B.2E)
============================================================
ROUTING_DECISION: ALL_OLLAMA
ACTIVE_MODEL: qwen2.5:1.5b
PROVIDER: ollama (http://localhost:11434)
CONFIDENCE: HIGH
PRODUCTION_ROUTING_CHANGED: NO
OPENROUTER_ROUTING_ACTIVE: NO
============================================================
```

### Decision Justification
1. **Quality**: Local Ollama (`qwen2.5:1.5b`) achieves the highest overall answer quality (**4.07 / 5.0**) compared to OpenRouter Model B (**3.76**) and OpenRouter Model A (**3.59**).
2. **Speed**: Local Ollama delivers a mean latency of **3.00s** (median **2.84s**), whereas OpenRouter Model B takes **12.64s** (4.2x slower) and OpenRouter Model A takes **39.73s** (13.2x slower).
3. **Reliability**: Eliminates API key dependencies, network latency spikes, rate limits, and upstream provider volatility.

---

## 19. Controls, Invariants & Guidance for Future AI Agents

### Frozen System Assets
- **Retrieval Pipeline**: BGE-M3 vector + BM25 + candidate filtering + `bge-reranker-v2-m3` reranker is **FROZEN**.
- **Vector Store Index**: `data/vector_store` (18 chunks, 10 documents) is **FROZEN**.
- **System Prompts**: Grounding prompt in `app/rag/llm.py` is **FROZEN**.
- **Production Routing**: `ROUTING_DECISION: ALL_OLLAMA` (`qwen2.5:1.5b`) is **FROZEN**.

### Next Required Steps
1. **Corpus Expansion**: If expanding domain knowledge beyond operating systems, ingest new text documents through `POST /documents/upload`.
2. **Paid Model Benchmark**: If evaluating paid commercial models (e.g. Claude 3.5 Sonnet, GPT-4o), reuse the Phase 5B.2B frozen context framework (`phase5b2_frozen_context_manifest.json`) without altering production code until evaluation completes.

### Mandatory Rules for Future AI Agents
> [!IMPORTANT]
> 1. **Do NOT Modify Candidate Document Scoping**: The document-scope filtering logic in `app/rag/retriever.py` enforces the critical invariant `FINAL_CONTEXT.documents ⊆ REQUESTED_DOCUMENTS`. Never remove or weaken explicit candidate filtering.
> 2. **Do NOT Switch Production Routing to OpenRouter Free Tier**: Empirical benchmark Phase 5B.2E proves OpenRouter free tier candidates (`nemotron-3.5-lightning` and `nemotron-nano-9b`) perform worse in quality while increasing latency by 400%–1300%.
> 3. **Do NOT Invent or Normalize Metrics**: All reported quality scores, latencies, and token counts must originate directly from verified raw output JSONL files (`phase5b2_raw_model_outputs.jsonl` and `phase5b2_raw_judge_results.jsonl`).
> 4. **Do NOT Make Production Code Changes During Evaluation**: Always maintain a strict separation between read-only evaluation scripts (`scratch/`) and production application code (`app/`).
