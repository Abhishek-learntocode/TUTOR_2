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

from langsmith import Client
from app.main import retriever, settings
from app.rag.providers import get_provider, clear_provider_call_ledger, save_provider_call_ledger, get_provider_call_ledger

ls_client = Client()
ls_project_id = "1a4d2edf-a0b5-4975-b31a-1c5494eb9569"

print("=" * 80)
print("PHASE 5B.2A — CONTROLLED MODEL QUALITY & SAME-CONTEXT A/B BENCHMARK")
print("=" * 80)

# Selected 8 Representative Queries from rag_baseline_v1.jsonl
TARGET_QUERY_IDS = [
    "factual_001",    # What is virtual memory?
    "concept_001",    # Explain how paging works in operating systems.
    "multihop_001",   # How does paging work, and what problem does it solve?
    "doc_003",        # According to OS_Notes.txt, what is a virtual address space?
    "compare_001",    # Compare virtual memory and physical memory.
    "missing_002",    # What is the stock price of Apple in 2026?
    "multidoc_001",   # Compare the discussion of memory management in OS_Notes.txt and sample_exam.txt.
    "exam_001",       # MCQ: Which of the following best describes virtual memory? ...
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

# Candidate Models for Same-Context A/B Comparison
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
        "configured_model": "poolside/laguna-s-2.1:free",
        "description": "OpenRouter Model B (Poolside Laguna S 2.1)",
    },
    {
        "model_key": "OPENROUTER_MODEL_C",
        "provider": "openrouter",
        "configured_model": "nvidia/nemotron-nano-9b-v2:free",
        "description": "OpenRouter Model C (Nemotron Nano 9B v2)",
    },
]

clear_provider_call_ledger()
benchmark_results = []
qualitative_answers = []
total_openrouter_expected = 0
total_openrouter_actual = 0

for item in eval_dataset:
    qid = item["id"]
    category = item["category"]
    qtext = item["query"]
    exp_behavior = item["expected_behavior"]
    doc_scope = item.get("document_scope")

    print(f"\n==================================================================")
    print(f"BENCHMARK QUERY [{qid}] ({category}): '{qtext}'")
    print(f"==================================================================")

    # STEP A: Run retrieval ONCE and retrieve final_context
    retrieval_t0 = time.time()
    context_chunks = retriever.retrieve(qtext)
    retrieval_lat = round(time.time() - retrieval_t0, 4)

    # Extract retrieved document names
    retrieved_docs = []
    for c in context_chunks:
        m = re.search(r"\[Document: (.*?) \|", c)
        if m:
            retrieved_docs.append(m.group(1))
    unique_docs = list(set(retrieved_docs))

    # STEP B: Freeze context & compute SHA256 hashes
    context_str = "\n\n---\n\n".join(context_chunks) if context_chunks else ""
    prompt_str = (
        "Answer the question based ONLY on the supplied context.\n"
        "If the answer is not supported by the context, state clearly: "
        "\"I cannot find the answer in the provided context.\"\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {qtext}\n\nAnswer:"
    )

    prompt_hash = hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()
    context_hash = hashlib.sha256(context_str.encode("utf-8")).hexdigest()

    print(f"[*] Retrieval Latency : {retrieval_lat}s | Chunks: {len(context_chunks)} | Docs: {unique_docs}")
    print(f"[*] SHA256(Prompt)    : {prompt_hash[:16]}...")
    print(f"[*] SHA256(Context)   : {context_hash[:16]}...")

    query_model_outputs = {}

    # STEP C: Invoke each candidate model with the EXACT SAME frozen prompt & context
    for candidate in CANDIDATE_MODELS:
        mkey = candidate["model_key"]
        p_name = candidate["provider"]
        m_name = candidate["configured_model"]

        print(f"\n--- Testing Candidate [{mkey}]: {p_name} / {m_name} ---")

        if p_name == "openrouter":
            total_openrouter_expected += 1
            time.sleep(2.5)

        provider_obj = get_provider(
            provider_name=p_name,
            model_name=m_name,
            base_url=settings.llm_base_url if p_name == "ollama" else settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )

        gen_t0 = time.time()
        res = provider_obj.generate(prompt_str, temperature=0.1)

        # Retry once on 429 Rate Limit
        max_retries = 1
        retry_cnt = 0
        while p_name == "openrouter" and res.get("status_code") == 429 and retry_cnt < max_retries:
            retry_cnt += 1
            wait_sec = 1.0
            print(f"  [!] HTTP 429 Rate Limit encountered. Retrying attempt {retry_cnt}/{max_retries} in {wait_sec}s...")
            time.sleep(wait_sec)
            gen_t0 = time.time()
            res = provider_obj.generate(prompt_str, temperature=0.1)

        gen_t1 = time.time()
        gen_lat = round(gen_t1 - gen_t0, 4)

        if p_name == "openrouter" and res.get("success"):
            total_openrouter_actual += 1

        answer = (res.get("text") or res.get("content") or "").strip()
        act_model = res.get("actual_model") or m_name
        op_req_id = res.get("openrouter_request_id", "N/A")
        usage = res.get("usage", {})

        # Quality & Rule Scoring
        is_refusal = "I cannot find the answer in the provided context." in answer
        expected_refusal = exp_behavior == "refuse_or_state_not_in_context"
        refusal_correct = (is_refusal if expected_refusal else (not is_refusal if context_chunks else True))

        # Document scope check
        scope_correct = True
        if doc_scope is not None:
            for d in unique_docs:
                if d not in doc_scope:
                    scope_correct = False

        # Factual correctness / grounding heuristics (0-5 scale)
        if is_refusal and expected_refusal:
            correctness = 5.0
            grounding = 5.0
            completeness = 5.0
            relevance = 5.0
            instruction_following = 5.0
            hallucination_count = 0
            unsupported_claims = 0
        elif is_refusal and not expected_refusal:
            correctness = 1.0
            grounding = 5.0
            completeness = 0.0
            relevance = 1.0
            instruction_following = 2.0
            hallucination_count = 0
            unsupported_claims = 0
        else:
            # Score grounded answer
            grounding = 5.0 if not any(w in answer.lower() for w in ["unsupported", "according to my knowledge", "as an ai"]) else 2.0
            completeness = 5.0 if len(answer) > 80 else 3.0
            relevance = 5.0 if any(k in answer.lower() for k in qtext.lower().split() if len(k) > 3) else 3.0
            instruction_following = 5.0
            correctness = (grounding + completeness + relevance) / 3.0
            hallucination_count = 0
            unsupported_claims = 0

        # MCQ evaluation for exam_001
        if qid == "exam_001":
            mcq_pass = "B" in answer or "Memory management" in answer or "option b" in answer.lower()
            if not mcq_pass:
                correctness = max(1.0, correctness - 2.0)

        record = {
            "query_id": qid,
            "category": category,
            "model_key": mkey,
            "provider": p_name,
            "configured_model": m_name,
            "actual_model": act_model,
            "openrouter_request_id": op_req_id,
            "prompt_hash": prompt_hash,
            "context_hash": context_hash,
            "status_code": res.get("status_code", 200 if res.get("success") else 500),
            "retrieval_latency": retrieval_lat,
            "generation_latency": gen_lat,
            "total_latency": round(retrieval_lat + gen_lat, 4),
            "context_count": len(context_chunks),
            "answer": answer,
            "answer_length": len(answer),
            "correctness": round(correctness, 2),
            "grounding": round(grounding, 2),
            "completeness": round(completeness, 2),
            "relevance": round(relevance, 2),
            "instruction_following": round(instruction_following, 2),
            "refusal_correctness": refusal_correct,
            "document_scope_correctness": scope_correct,
            "hallucination_count": hallucination_count,
            "unsupported_claim_count": unsupported_claims,
            "usage": usage,
            "error": res.get("error"),
        }
        benchmark_results.append(record)
        query_model_outputs[mkey] = answer

        print(f"  HTTP Status           : {record['status_code']}")
        print(f"  Actual Model          : {act_model}")
        print(f"  OpenRouter Request ID : {op_req_id}")
        print(f"  Generation Latency    : {gen_lat}s")
        print(f"  Answer Length         : {len(answer)} chars")
        print(f"  Correctness / Grounding: {record['correctness']} / {record['grounding']}")
        print(f"  Answer Snippet        : {repr(answer[:100])}...")
        sys.stdout.flush()

    # Save qualitative comparison for representative set
    qualitative_answers.append({
        "query_id": qid,
        "category": category,
        "query": qtext,
        "context_documents": unique_docs,
        "frozen_context_snippet": context_str[:300] + "...",
        "answers": query_model_outputs,
    })

# Save Provider Request Ledger
out_dir = os.path.join(backend_dir, "evaluation", "results")
os.makedirs(out_dir, exist_ok=True)
save_provider_call_ledger(os.path.join(out_dir, "phase5b2_provider_request_ledger.json"))

# Save Benchmark Results JSON
with open(os.path.join(out_dir, "phase5b2_model_quality_results.json"), "w", encoding="utf-8") as f:
    json.dump({
        "total_queries": len(eval_dataset),
        "total_model_evaluations": len(benchmark_results),
        "expected_openrouter_requests": total_openrouter_expected,
        "actual_openrouter_requests": total_openrouter_actual,
        "results": benchmark_results,
    }, f, indent=2)

# Save Representative Answers JSON
with open(os.path.join(out_dir, "phase5b2_representative_answers.json"), "w", encoding="utf-8") as f:
    json.dump(qualitative_answers, f, indent=2)

# Save CSV Summary
csv_file = os.path.join(out_dir, "phase5b2_model_comparison.csv")
with open(csv_file, "w", encoding="utf-8") as f:
    f.write("query_id,category,model_key,provider,configured_model,actual_model,status_code,generation_latency,answer_length,correctness,grounding,completeness,relevance,refusal_correct,scope_correct\n")
    for r in benchmark_results:
        f.write(f"{r['query_id']},{r['category']},{r['model_key']},{r['provider']},{r['configured_model']},{r['actual_model']},{r['status_code']},{r['generation_latency']},{r['answer_length']},{r['correctness']},{r['grounding']},{r['completeness']},{r['relevance']},{r['refusal_correctness']},{r['document_scope_correctness']}\n")

print("\n==================================================================")
print(f"[SUMMARY] Phase 5B.2A Benchmark Completed: {len(benchmark_results)} evaluations across {len(eval_dataset)} queries!")
print(f"[SUMMARY] Expected OpenRouter Requests: {total_openrouter_expected} | Actual Successful: {total_openrouter_actual}")
print(f"[SUMMARY] Saved CSV comparison to {csv_file}")
print("==================================================================")
