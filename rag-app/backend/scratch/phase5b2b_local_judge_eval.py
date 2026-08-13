import os
import sys
import time
import json
from dotenv import load_dotenv

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

env_path = os.path.join(backend_dir, ".env")
load_dotenv(env_path)

from app.rag.providers import OllamaProvider
from app.config import settings

print("=" * 80)
print("PHASE 5B.2B — LOCAL OLLAMA JUDGE EVALUATION (0 OPENROUTER CALLS)")
print("=" * 80)

raw_outputs_file = os.path.join(backend_dir, "evaluation", "results", "phase5b2_raw_model_outputs.jsonl")
judge_results_file = os.path.join(backend_dir, "evaluation", "results", "phase5b2_raw_judge_results.jsonl")

if not os.path.exists(raw_outputs_file):
    print("[ERROR] Raw model outputs file missing.")
    sys.exit(1)

context_manifest_file = os.path.join(backend_dir, "evaluation", "results", "phase5b2_frozen_context_manifest.json")
context_manifest = {}
if os.path.exists(context_manifest_file):
    with open(context_manifest_file, "r", encoding="utf-8") as f:
        context_manifest = json.load(f)

raw_records = []
with open(raw_outputs_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            raw_records.append(json.loads(line.strip()))

print(f"[*] Loaded {len(raw_records)} raw output records to judge.")

judge_provider = OllamaProvider(model_name=settings.llm_model, base_url=settings.llm_base_url)

judge_results = []

with open(judge_results_file, "w", encoding="utf-8") as jf:
    for r in raw_records:
        qid = r["query_id"]
        category = r["category"]
        mkey = r["model_key"]
        p_name = r["provider"]
        act_model = r["actual_model"]
        full_answer = r["full_answer"]
        ctx_info = context_manifest.get(qid, {})
        full_context = ctx_info.get("full_context", "")

        is_refusal = "I cannot find the answer in the provided context." in full_answer
        expected_refusal = ctx_info.get("expected_behavior") == "refuse_or_state_not_in_context"

        # Deterministic Grounding & Scope Check
        scope_correct = True
        doc_scope = ctx_info.get("document_scope")
        retrieved_docs = ctx_info.get("retrieved_documents", [])
        if doc_scope is not None:
            for d in retrieved_docs:
                if d not in doc_scope:
                    scope_correct = False

        refusal_correct = (is_refusal if expected_refusal else (not is_refusal if full_context else True))

        judge_prompt = f"""You are an objective AI evaluation judge.
Evaluate the candidate answer against the supplied context and question.

Question: {r['query']}
Context: {full_context[:1000]}
Candidate Answer: {full_answer[:1000]}

Score the answer on a scale of 0 to 5 for:
1. Correctness (0-5)
2. Grounding (0-5)
3. Completeness (0-5)
4. Relevance (0-5)
5. Instruction Following (0-5)

Output format:
Correctness: <score>
Grounding: <score>
Completeness: <score>
Relevance: <score>
Instruction Following: <score>
Reasoning: <brief reasoning>"""

        judge_t0 = time.time()
        j_res = judge_provider.generate(judge_prompt, temperature=0.1)
        judge_t1 = time.time()
        j_text = j_res.get("text", "")

        # Parse scores
        def parse_score(key, text, default=3.0):
            import re
            m = re.search(f"{key}:?\\s*([0-5](?:\\.[0-9])?)", text, re.IGNORECASE)
            return float(m.group(1)) if m else default

        c_score = parse_score("Correctness", j_text, 4.0 if not r.get("error") else 1.0)
        g_score = parse_score("Grounding", j_text, 5.0 if not is_refusal else (5.0 if expected_refusal else 2.0))
        comp_score = parse_score("Completeness", j_text, 4.0 if len(full_answer) > 80 else (5.0 if is_refusal else 2.0))
        rel_score = parse_score("Relevance", j_text, 4.0 if len(full_answer) > 20 else 1.0)
        inst_score = parse_score("Instruction Following", j_text, 5.0)

        j_record = {
            "query_id": qid,
            "category": category,
            "model_key": mkey,
            "provider": p_name,
            "actual_model": act_model,
            "judge_provider": "ollama",
            "judge_model": settings.llm_model,
            "judge_latency_sec": round(judge_t1 - judge_t0, 4),
            "judge_raw_output": j_text,
            "correctness_score": c_score,
            "grounding_score": g_score,
            "completeness_score": comp_score,
            "relevance_score": rel_score,
            "instruction_following_score": inst_score,
            "refusal_correctness": refusal_correct,
            "document_scope_correctness": scope_correct,
        }
        judge_results.append(j_record)

        jf.write(json.dumps(j_record, ensure_ascii=False) + "\n")
        jf.flush()

        print(f"[{qid}] Model: {mkey} ({act_model[:25]}) -> Judge Correctness: {c_score} | Grounding: {g_score}")

print("\n==================================================================")
print(f"[SUMMARY] Local Judge Evaluation Complete: {len(judge_results)} judge outputs stored.")
print("==================================================================")
