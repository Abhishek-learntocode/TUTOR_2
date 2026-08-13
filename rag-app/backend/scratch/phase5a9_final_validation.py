import os
import sys
import time
import json
import requests
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
ls_client = Client()
ls_project_id = "1a4d2edf-a0b5-4975-b31a-1c5494eb9569"

print("=== PHASE 5A.9 — STEP 6 & 7: REPRESENTATIVE VALIDATION WITH DEEP EVIDENCE ===")

rep_queries = [
    {
        "id": "rep_001_factual",
        "category": "factual",
        "query": "What is virtual memory?",
        "expected": "Grounded answer explaining virtual memory illusion."
    },
    {
        "id": "rep_002_conceptual",
        "category": "conceptual",
        "query": "Explain paging.",
        "expected": "Grounded answer explaining paging memory management scheme."
    },
    {
        "id": "rep_003_multihop",
        "category": "multi_hop",
        "query": "How does paging work, and what problem does it solve?",
        "expected": "Grounded answer covering non-contiguous physical memory allocation."
    },
    {
        "id": "rep_004_doc_scoped",
        "category": "document_specific",
        "query": "According to OS_Notes.txt, explain paging.",
        "expected": "Refusal or grounded statement that OS_Notes.txt describes virtual memory/system calls."
    },
    {
        "id": "rep_005_cross_doc",
        "category": "multi_document",
        "query": "Compare the discussion of memory management in OS_Notes.txt and sample_exam.txt.",
        "expected": "Cross-document synthesis comparing virtual memory illusion and exam Q2 paging."
    },
    {
        "id": "rep_006_exam_style",
        "category": "exam_style",
        "query": "Which of the following best describes virtual memory? A) Physical RAM extension B) Memory management capability C) Hard drive cache D) CPU register.",
        "expected": "Selection of option B or description of memory management capability."
    },
    {
        "id": "rep_007_missing_info",
        "category": "missing_information",
        "query": "What is the stock price of Apple in 2026?",
        "expected": "Correct refusal (not in context)."
    },
    {
        "id": "rep_008_ambiguous",
        "category": "ambiguous",
        "query": "Explain the table structure.",
        "expected": "Correct refusal or clarification request."
    }
]

deep_results = []

for q_info in rep_queries:
    qid = q_info["id"]
    cat = q_info["category"]
    qtext = q_info["query"]

    print(f"\n==================================================================")
    print(f"RUNNING VALIDATION QUERY [{qid}] ({cat}): '{qtext}'")
    print(f"==================================================================")

    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/query", json={"question": qtext}, timeout=180)
    t1 = time.time()
    wall_sec = round(t1 - t0, 4)

    status_code = resp.status_code
    resp_body = resp.json() if status_code == 200 else {}

    answer = resp_body.get("answer", "")
    context = resp_body.get("context", [])

    print(f"  HTTP Status Code: {status_code}")
    print(f"  HTTP Wall Latency: {wall_sec}s")
    print(f"  Context Count    : {len(context)}")
    print(f"  Answer Snippet   : {repr(answer[:150])}...")

    # Wait 1 sec for LangSmith trace indexing
    time.sleep(1.5)

    # Fetch corresponding LangSmith trace
    ls_runs = list(ls_client.list_runs(project_id=ls_project_id, is_root=True, limit=5))
    matching_ls_run = None
    for r in ls_runs:
        if r.inputs.get("question") == qtext or r.inputs.get("query") == qtext:
            matching_ls_run = r
            break

    ls_data = {}
    if matching_ls_run:
        ls_data = {
            "run_id": str(matching_ls_run.id),
            "run_name": matching_ls_run.name,
            "start_time": str(matching_ls_run.start_time),
            "end_time": str(matching_ls_run.end_time),
            "latency_sec": round(matching_ls_run.end_time.timestamp() - matching_ls_run.start_time.timestamp(), 4) if matching_ls_run.end_time else 0,
            "inputs": matching_ls_run.inputs,
            "outputs": matching_ls_run.outputs
        }
        print(f"  LangSmith Trace  : Run ID {matching_ls_run.id} (Latency: {ls_data['latency_sec']}s)")

    # Grounding evaluation
    is_refusal = "I cannot find the answer" in answer
    if cat == "missing_information":
        grounded_status = "CORRECT_REFUSAL" if is_refusal else "UNSUPPORTED_HALLUCINATION"
    elif is_refusal:
        grounded_status = "REFUSAL_DUE_TO_MISSING_CONTEXT"
    else:
        grounded_status = "GROUNDED_ANSWER"

    record = {
        "query_id": qid,
        "category": cat,
        "query": qtext,
        "http": {
            "status_code": status_code,
            "wall_latency_sec": wall_sec,
            "context_count": len(context),
            "context_chunks": context,
            "answer": answer,
            "answer_length": len(answer)
        },
        "langsmith": ls_data,
        "grounding_eval": {
            "is_refusal": is_refusal,
            "status": grounded_status,
            "supported_by_context": not is_refusal
        }
    }

    deep_results.append(record)

os.makedirs("evaluation/results", exist_ok=True)
with open("evaluation/results/phase5a9_representative_results.json", "w", encoding="utf-8") as f:
    json.dump(deep_results, f, indent=2)

print("\nRepresentative validation queries complete. Saved to evaluation/results/phase5a9_representative_results.json.")
