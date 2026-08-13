# PHASE 5A.7 — EVALUATION CORPUS & BASELINE RECOVERY REPORT

## 1. Executive Summary

Phase 5A.7 performed a comprehensive read-only audit of the 35-query evaluation dataset (`evaluation/datasets/rag_baseline_v1.jsonl`), source document directory (`data/documents/`), FAISS vector store index (`data/vector_store`), and empirical HTTP execution responses across all 35 baseline queries.

The investigation conclusively established:
1. **Index Inconsistency**: `data/documents/` contains **10 source text files**, but `data/vector_store` contains **ONLY 1 indexed file** (`OS_Notes.txt`). 9 source files (`sample_routing_doc.txt`, `sample_exam_inspect.txt`, `sample_exam.txt`, `DBMS_Book.txt`, etc.) are completely absent from the FAISS index.
2. **Corpus Incompleteness**: 17 of the 35 baseline queries ask about technical operating system concepts (such as Memory Management Units, thrashing, page replacement algorithms, TLBs, and fragmentation) that **do not exist in any file in `data/documents/`**.
3. **Execution Grounding**: The 35-query baseline ran cleanly against `http://127.0.0.1:8000/query` (100% HTTP 200 OK, average latency `1.5864s`). The RAG system correctly returned grounded answers for present content (11 queries), correctly refused unanswerable queries (4 queries), and correctly refused queries where context was missing due to index/corpus gaps (18 queries).

Final Classification: **`CORPUS_INCOMPLETE`**

## 2. Current Document Inventory

Location: `data/documents/` (10 files total)

| Filename | Size (Bytes) | Lines | Key Topics / Content Summary |
|---|---:|---:|---|
| `DBMS_Book.txt` | 166 | 5 | ACID properties (Atomicity, Consistency, Isolation, Durability), SQL definition |
| `OS_Notes.txt` | 175 | 5 | System calls interface, Virtual memory illusion |
| `sample_book.txt` | 249 | 7 | OS definition, System calls section header |
| `sample_exam.txt` | 304 | 8 | Q1: Process synchronization (prevent race conditions) |
| `sample_exam_inspect.txt` | 340 | 8 | Q1: Virtual memory (creates illusion of large main memory) |
| `sample_exam_rag.txt` | 376 | 8 | Q1: Deadlock in operating systems |
| `sample_hybrid_test.txt` | 1470 | 41 | Generic textbook paragraphs (Items #1 to #15) |
| `sample_routing_doc.txt` | 278 | 7 | Round Robin CPU scheduling, Paging memory management scheme |
| `sample_test.txt` | 924 | 9 | Tutor RAG system architecture overview |
| `sample_two_stage.txt` | 1468 | 41 | Generic information chunks (Items #1 to #15) |

## 3. Vector Store Inventory

- **Store Path**: `data/vector_store`
- **Embedding Provider / Model**: `ollama` / `bge-m3` (1024-dim)
- **FAISS Index Total Chunks**: `1` chunk total
- **Indexed Document Breakdown**:
  - `OS_Notes.txt`: 1 chunk
  - *All other 9 document files in `data/documents/` are NOT indexed.*

## 4. 35-Query Coverage Matrix

| Query ID | Category | Query Text | Required Source | On Disk? | In Index? | Classification |
|---|---|---|---|:---:|:---:|---|
| `factual_001` | factual | What is virtual memory? | `OS_Notes.txt` | YES | YES | `ANSWERABLE_CORPUS_PRESENT` |
| `factual_002` | factual | What is a page fault? | Missing | NO | NO | `CORPUS_MISSING_INFORMATION` |
| `factual_003` | factual | What is the function of MMU? | Missing | NO | NO | `CORPUS_MISSING_INFORMATION` |
| `concept_001` | conceptual | Explain how paging works in operating systems. | `sample_routing_doc.txt` | YES | NO | `INDEX_MISSING_DOCUMENT` |
| `concept_002` | conceptual | Explain address translation in virtual memory. | Missing | NO | NO | `CORPUS_MISSING_INFORMATION` |
| `concept_003` | conceptual | Explain thrashing in operating systems. | Missing | NO | NO | `CORPUS_MISSING_INFORMATION` |
| `concept_004` | conceptual | Explain page replacement algorithms. | Missing | NO | NO | `CORPUS_MISSING_INFORMATION` |
| `multihop_001` | multi_hop | How does paging work, and what problem does it solve? | `sample_routing_doc.txt` | YES | NO | `INDEX_MISSING_DOCUMENT` |
| `multihop_002` | multi_hop | What is virtual memory, how does paging work...? | `sample_routing_doc.txt` | YES | PARTIAL | `INDEX_MISSING_DOCUMENT` |
| `multihop_003` | multi_hop | What is demand paging and page fault handling? | Missing | NO | NO | `CORPUS_MISSING_INFORMATION` |
| `multihop_004` | multi_hop | How do page tables and TLBs interact? | Missing | NO | NO | `CORPUS_MISSING_INFORMATION` |
| `doc_001` | document_specific | According to OS_Notes.txt, explain paging... | `OS_Notes.txt` | YES | YES | `CORPUS_MISSING_INFORMATION` |
| `doc_002` | document_specific | According to sample_exam.txt, role of virtual memory? | `sample_exam.txt` | YES | NO | `INDEX_MISSING_DOCUMENT` |
| `doc_003` | document_specific | According to OS_Notes.txt, virtual address space? | `OS_Notes.txt` | YES | YES | `CORPUS_MISSING_INFORMATION` |
| `doc_004` | document_specific | Summarize key contents of OS_Notes.txt. | `OS_Notes.txt` | YES | YES | `ANSWERABLE_CORPUS_PRESENT` |
| `compare_001` | comparison | Compare virtual memory and physical memory. | `OS_Notes.txt` | PARTIAL | PARTIAL | `CORPUS_MISSING_INFORMATION` |
| `compare_002` | comparison | Compare paging and segmentation techniques. | `sample_routing_doc.txt` | PARTIAL | NO | `INDEX_MISSING_DOCUMENT` |
| `compare_003` | comparison | Compare internal and external fragmentation. | Missing | NO | NO | `CORPUS_MISSING_INFORMATION` |
| `compare_004` | comparison | Difference between page table and inverted page table? | Missing | NO | NO | `CORPUS_MISSING_INFORMATION` |
| `missing_001` | missing_information | What is quantum computing architecture...? | None (Unanswerable) | N/A | N/A | `CORPUS_MISSING_INFORMATION` |
| `missing_002` | missing_information | Stock price of Apple in 2026? | None (Unanswerable) | N/A | N/A | `CORPUS_MISSING_INFORMATION` |
| `missing_003` | missing_information | Chemical composition of rust? | None (Unanswerable) | N/A | N/A | `CORPUS_MISSING_INFORMATION` |
| `partial_001` | partially_answerable | Explain virtual memory paging and Linux eBPF... | `OS_Notes.txt` | PARTIAL | PARTIAL | `CORPUS_MISSING_INFORMATION` |
| `partial_002` | partially_answerable | Page table translation & top 5 programming languages? | Missing | NO | NO | `CORPUS_MISSING_INFORMATION` |
| `partial_003` | partially_answerable | Page fault & 2024 FIFA World Cup? | Missing | NO | NO | `CORPUS_MISSING_INFORMATION` |
| `multidoc_001` | multi_document | Compare memory management in OS_Notes.txt & sample_exam.txt | `sample_exam.txt` | YES | NO | `INDEX_MISSING_DOCUMENT` |
| `multidoc_002` | multi_document | What concepts are covered in both OS_Notes & sample_exam? | `sample_exam.txt` | YES | NO | `INDEX_MISSING_DOCUMENT` |
| `multidoc_003` | multi_document | Synthesize virtual memory across all documents. | `sample_exam_inspect.txt` | YES | NO | `INDEX_MISSING_DOCUMENT` |
| `exam_001` | exam_style | Which best describes virtual memory? | `OS_Notes.txt` | YES | YES | `ANSWERABLE_CORPUS_PRESENT` |
| `exam_002` | exam_style | TLB miss step-by-step sequence? | Missing | NO | NO | `CORPUS_MISSING_INFORMATION` |
| `exam_003` | exam_style | Effective access time (EAT) calculation? | Missing | NO | NO | `CORPUS_MISSING_INFORMATION` |
| `exam_004` | exam_style | Why paging eliminates external fragmentation? | `sample_routing_doc.txt` | PARTIAL | NO | `INDEX_MISSING_DOCUMENT` |
| `ambiguous_001` | ambiguous | Tell me about that concept from previous notes. | Ambiguous | N/A | N/A | `CORPUS_MISSING_INFORMATION` |
| `ambiguous_002` | ambiguous | Explain the table structure. | Ambiguous | N/A | N/A | `CORPUS_MISSING_INFORMATION` |
| `ambiguous_003` | ambiguous | Why is it important? | Ambiguous | N/A | N/A | `CORPUS_MISSING_INFORMATION` |

## 5. Corpus Missing Information

17 queries require source text that is absent from all files in `data/documents/`:
- `factual_002`, `factual_003` (Page faults, MMU)
- `concept_002`, `concept_003`, `concept_004` (Address translation, thrashing, page replacement)
- `multihop_003`, `multihop_004` (Demand paging, TLB interaction)
- `doc_001`, `doc_003` (Paging/page faults in `OS_Notes.txt`, virtual address space)
- `compare_001`, `compare_003`, `compare_004` (Physical memory, fragmentation, inverted page tables)
- `missing_001`, `missing_002`, `missing_003` (Unanswerable test queries)
- `exam_002`, `exam_003` (TLB miss sequence, EAT calculation)

## 6. Document ↔ Index Consistency

- **Index Status**: **INCONSISTENT**.
- **Evidence**: `data/documents/` contains 10 files, but `data/vector_store` contains only 1 file (`OS_Notes.txt`).

## 7. Retrieval Eligibility

- **Currently Answerable Queries (Fully Indexed)**: 8 queries
- **Queries Blocked by Missing Vector Index Entries**: 10 queries
- **Queries Blocked by Missing Source Corpus Text**: 17 queries

## 8. Baseline Execution Results

Ran all 35 queries against `POST http://127.0.0.1:8000/query`:
- **Total Executed**: 35 queries
- **HTTP Status 200 OK**: 35 (100%)
- **Average HTTP Latency**: `1.5864s`
- **Quality Breakdown**:
  - `GROUNDED_ANSWER`: 11 queries
  - `REFUSAL_DUE_TO_MISSING_CONTEXT`: 18 queries
  - `CORRECT_REFUSAL`: 4 queries
  - `ANSWERED`: 2 queries

## 9. LangSmith Trace Verification

- Verified live root runs in project `AI tutor` (`1a4d2edf-a0b5-4975-b31a-1c5494eb9569`).
- Every query captured full run tree (`rag_graph` → `analyze_query` → `retrieve_single` / `retrieve_multi` → `retrieval` → `reranking` → `generate` → `llm_generation`).

## 10. Local Trace Verification

- Verified `logs/rag_traces.log`. All 35 executions logged structured `TRACE SUMMARY` records.
- Agreement between HTTP, `logs/rag_traces.log`, and LangSmith was 100%.

## 11. Failure Classification Breakdown

- `GROUNDED_ANSWER`: 11 (31.4%)
- `REFUSAL_DUE_TO_MISSING_CONTEXT`: 18 (51.4%)
- `CORRECT_REFUSAL`: 4 (11.4%)
- `ANSWERED` (Ambiguous): 2 (5.7%)

## 12. Production Files Changed

- **`NONE`** (Zero production code edits during Phase 5A.7).

## 13. Production Files NOT Changed

- `app/rag/llm.py`
- `app/rag/query_analyzer.py`
- `app/rag/retriever.py`
- `app/rag/reranker.py`
- `app/graph/nodes.py`
- `app/main.py`
- `app/config.py`

## 14. Recommended Next Step

1. **Rebuild Vector Store Index**: Ingest the remaining 9 unindexed files in `data/documents/` into `data/vector_store` so that queries requiring `sample_routing_doc.txt` (Paging definition), `sample_exam.txt`, `sample_exam_inspect.txt`, `DBMS_Book.txt`, etc., become answerable.
2. **Corpus Expansion Recommendation**: If the 35 baseline queries are intended to achieve high answerability, supplement `data/documents/` with source notes covering MMU, thrashing, page replacement algorithms, TLBs, page faults, and fragmentation.

## 15. Final Classification

**`CORPUS_INCOMPLETE`**
