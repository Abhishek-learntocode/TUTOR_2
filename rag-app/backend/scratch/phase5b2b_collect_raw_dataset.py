import os
import sys
import time
import json
import re
import hashlib
from dotenv import load_dotenv

# Ensure backend root is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

env_path = os.path.join(backend_dir, ".env")
load_dotenv(env_path)

from app.main import retriever, settings
from app.rag.providers import get_provider

print("=" * 80)
print("PHASE 5B.2B — CONTROLLED MODEL EVALUATION DATA COLLECTION")
print("=" * 80)

# Strict Budget Management
OPENROUTER_BUDGET = 40
OPENROUTER_USED = 0

# Selected 15 Representative Queries from rag_baseline_v1.jsonl
TARGET_QUERY_IDS = [
    "factual_001",    # What is virtual memory?
    "factual_002",    # What is a page fault?
    "factual_003",    # What is the function of the Memory Management Unit (MMU)?
    "concept_001",    # Explain how paging works in operating systems.
    "concept_002",    # Explain how address translation is performed in virtual memory systems.
    "concept_003",    # Explain the concept of thrashing in operating systems.
    "multihop_001",   # How does paging work, and what problem does it solve?
    "multihop_002",   # What is virtual memory, how does paging work, and why are page faults needed?
    "multihop_004",   # How do page tables and TLBs interact to accelerate memory reference translation?
    "doc_001",        # According to OS_Notes.txt, explain paging and page faults.
    "doc_003",        # According to OS_Notes.txt, what is a virtual address space?
    "compare_001",    # Compare virtual memory and physical memory.
    "multidoc_001",   # Compare the discussion of memory management in OS_Notes.txt and sample_exam.txt.
    "exam_001",       # MCQ: Which of the following best describes virtual memory? ...
    "missing_002",    # What is the stock price of Apple in 2026?
]

dataset_path = os.path.join(backend_dir, "evaluation", "datasets", "rag_baseline_v1.jsonl")
eval_dataset = []
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line.strip())
            if obj["id"] in TARGET_QUERY_IDS:
                eval_dataset.append(obj)

print(f"[*] Loaded {len(eval_dataset)} representative benchmark queries.")

# Candidate Models (Ollama Baseline + 2 Fixed OpenRouter Candidates)
CANDIDATE_MODELS = [
    {
        "model_key": "OLLAMA_BASELINE",
        "provider": "ollama",
        "configured_model": "qwen2.5:1.5b",
        "description": "Local Baseline (Qwen2.5 1.5B)",
    },
    {
        "model_key": "OPENROUTER_MODEL_A",
        "provider": "openrouter",
        "configured_model": "nvidia/nemotron-3.5-lightning:free",
        "description": "OpenRouter Model A (Nemotron 3.5 Lightning 1M)",
    },
    {
        "model_key": "OPENROUTER_MODEL_B",
        "provider": "openrouter",
        "configured_model": "nvidia/nemotron-nano-9b-v2:free",
        "description": "OpenRouter Model B (Nemotron Nano 9B v2)",
    },
]

out_dir = os.path.join(backend_dir, "evaluation", "results")
os.makedirs(out_dir, exist_ok=True)

raw_outputs_file = os.path.join(out_dir, "phase5b2_raw_model_outputs.jsonl")
context_manifest_file = os.path.join(out_dir, "phase5b2_frozen_context_manifest.json")
ledger_file = os.path.join(out_dir, "phase5b2_request_ledger.json")
metadata_file = os.path.join(out_dir, "phase5b2_model_metadata.json")
trace_file = os.path.join(out_dir, "phase5b2_trace_manifest.json")

# Write model metadata
with open(metadata_file, "w", encoding="utf-8") as f:
    json.dump({
        "collection_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidates": CANDIDATE_MODELS,
        "ollama_baseline": "qwen2.5:1.5b",
        "openrouter_models": ["nvidia/nemotron-3.5-lightning:free", "nvidia/nemotron-nano-9b-v2:free"],
    }, f, indent=2)

raw_output_records = []
context_manifest = {}
request_ledger = []
trace_manifest = []

openrouter_req_count = 0
openrouter_success_count = 0
openrouter_failed_count = 0

with open(raw_outputs_file, "w", encoding="utf-8") as raw_f:
    for item in eval_dataset:
        qid = item["id"]
        category = item["category"]
        qtext = item["query"]
        exp_behavior = item["expected_behavior"]
        doc_scope = item.get("document_scope")

        print(f"\n==================================================================")
        print(f"BENCHMARK QUERY [{qid}] ({category}): '{qtext}'")
        print(f"==================================================================")

        # 1. Execute retrieval ONCE
        retrieval_t0 = time.time()
        context_chunks = retriever.retrieve(qtext)
        retrieval_lat = round(time.time() - retrieval_t0, 4)

        retrieved_docs = []
        for c in context_chunks:
            m = re.search(r"\[Document: (.*?) \|", c)
            if m:
                retrieved_docs.append(m.group(1))
        unique_docs = list(set(retrieved_docs))

        # 2. Freeze context & compute SHA256 hashes
        context_str = "\n\n---\n\n".join(context_chunks) if context_chunks else ""
        prompt_str = (
            "Answer the question based ONLY on the supplied context.\n"
            "If the answer is not supported by the context, state clearly: "
            "\"I cannot find the answer in the provided context.\"\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {qtext}\n\nAnswer:"
        )

        prompt_sha256 = hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()
        context_sha256 = hashlib.sha256(context_str.encode("utf-8")).hexdigest()

        context_manifest[qid] = {
            "query_id": qid,
            "query": qtext,
            "category": category,
            "expected_behavior": exp_behavior,
            "document_scope": doc_scope,
            "retrieved_documents": unique_docs,
            "context_chunk_count": len(context_chunks),
            "retrieval_latency_sec": retrieval_lat,
            "prompt_sha256": prompt_sha256,
            "context_sha256": context_sha256,
            "full_context": context_str,
        }

        print(f"[*] Retrieval Latency : {retrieval_lat}s | Chunks: {len(context_chunks)} | Docs: {unique_docs}")
        print(f"[*] SHA256(Prompt)    : {prompt_sha256[:16]}...")
        print(f"[*] SHA256(Context)   : {context_sha256[:16]}...")

        # 3. Invoke candidate models
        for candidate in CANDIDATE_MODELS:
            mkey = candidate["model_key"]
            p_name = candidate["provider"]
            m_name = candidate["configured_model"]

            if p_name == "openrouter":
                if OPENROUTER_USED >= OPENROUTER_BUDGET:
                    print(f"  [STOP] HARD OPENROUTER BUDGET REACHED ({OPENROUTER_USED}/{OPENROUTER_BUDGET}). Skipping request.")
                    continue
                OPENROUTER_USED += 1
                openrouter_req_count += 1

                # Pace OpenRouter requests
                time.sleep(2.0)

            print(f"\n--- Model [{mkey}]: {p_name} / {m_name} ---")

            provider_obj = get_provider(
                provider_name=p_name,
                model_name=m_name,
                base_url=settings.llm_base_url if p_name == "ollama" else settings.openrouter_base_url,
                api_key=settings.openrouter_api_key,
            )

            gen_t0 = time.time()
            res = provider_obj.generate(prompt_str, temperature=0.1)

            # Retry once on 429 Rate Limit if budget allows
            if p_name == "openrouter" and res.get("status_code") == 429 and OPENROUTER_USED < OPENROUTER_BUDGET:
                print("  [!] HTTP 429 Rate Limit encountered. Pausing 3.0s before 1 retry...")
                time.sleep(3.0)
                gen_t0 = time.time()
                res = provider_obj.generate(prompt_str, temperature=0.1)

            gen_t1 = time.time()
            gen_lat = round(gen_t1 - gen_t0, 4)

            scode = res.get("status_code", 200 if res.get("success") else 500)
            if p_name == "openrouter":
                if res.get("success"):
                    openrouter_success_count += 1
                else:
                    openrouter_failed_count += 1

                ledger_entry = {
                    "request_number": openrouter_req_count,
                    "query_id": qid,
                    "model": m_name,
                    "actual_model": res.get("actual_model", m_name),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "success" if res.get("success") else "failed",
                    "status_code": scode,
                    "request_id": res.get("openrouter_request_id", "N/A"),
                    "latency_sec": gen_lat,
                    "usage": res.get("usage", {}),
                    "error": res.get("error"),
                }
                request_ledger.append(ledger_entry)

            full_answer = (res.get("text") or res.get("content") or "").strip()
            act_model = res.get("actual_model") or m_name
            op_req_id = res.get("openrouter_request_id", "N/A")
            usage = res.get("usage", {})

            raw_record = {
                "query_id": qid,
                "category": category,
                "query": qtext,
                "model_key": mkey,
                "provider": p_name,
                "requested_model": m_name,
                "actual_model": act_model,
                "request_id": op_req_id,
                "status_code": scode,
                "prompt_sha256": prompt_sha256,
                "context_sha256": context_sha256,
                "context_chunk_count": len(context_chunks),
                "context_documents": unique_docs,
                "retrieval_latency_sec": retrieval_lat,
                "generation_latency_sec": gen_lat,
                "wall_latency_sec": round(retrieval_lat + gen_lat, 4),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "full_answer": full_answer,
                "answer_length_chars": len(full_answer),
                "error": res.get("error"),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            raw_output_records.append(raw_record)

            # Write JSONL line immediately
            raw_f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
            raw_f.flush()

            trace_manifest.append({
                "query_id": qid,
                "model_key": mkey,
                "provider": p_name,
                "actual_model": act_model,
                "request_id": op_req_id,
                "status_code": scode,
                "latency_sec": gen_lat,
            })

            print(f"  HTTP Status           : {scode}")
            print(f"  Actual Model          : {act_model}")
            print(f"  Request ID            : {op_req_id}")
            print(f"  Generation Latency    : {gen_lat}s")
            print(f"  Full Answer Length    : {len(full_answer)} chars")
            print(f"  Answer Snippet        : {repr(full_answer[:80])}...")
            sys.stdout.flush()

# Save Context Manifest
with open(context_manifest_file, "w", encoding="utf-8") as f:
    json.dump(context_manifest, f, indent=2)

# Save Request Ledger
with open(ledger_file, "w", encoding="utf-8") as f:
    json.dump({
        "openrouter_requests_used": OPENROUTER_USED,
        "openrouter_requests_remaining": OPENROUTER_BUDGET - OPENROUTER_USED,
        "hard_limit": OPENROUTER_BUDGET,
        "successful_requests": openrouter_success_count,
        "failed_requests": openrouter_failed_count,
        "ledger": request_ledger,
    }, f, indent=2)

# Save Trace Manifest
with open(trace_file, "w", encoding="utf-8") as f:
    json.dump({"total_traces": len(trace_manifest), "traces": trace_manifest}, f, indent=2)

print("\n==================================================================")
print(f"[SUMMARY] Phase 5B.2B Raw Data Collection Complete!")
print(f"[SUMMARY] Total Raw Outputs Stored      : {len(raw_output_records)}")
print(f"[SUMMARY] OpenRouter Budget Hard Limit  : {OPENROUTER_BUDGET}")
print(f"[SUMMARY] OpenRouter Requests Used      : {OPENROUTER_USED}")
print(f"[SUMMARY] OpenRouter Requests Remaining : {OPENROUTER_BUDGET - OPENROUTER_USED}")
print(f"[SUMMARY] Budget Limit Respected       : {'PASS' if OPENROUTER_USED <= OPENROUTER_BUDGET else 'FAIL'}")
print("==================================================================")
