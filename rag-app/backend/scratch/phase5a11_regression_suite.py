import os
import sys
import time
import json
import requests
import re
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

ls_client = Client()
ls_project_id = "1a4d2edf-a0b5-4975-b31a-1c5494eb9569"
BASE_URL = "http://127.0.0.1:8000"

print("=== PHASE 5A.11 — REGRESSION TEST SUITE ===")

test_definitions = [
    {
        "test_id": "test_1_general_retrieval",
        "query": "What is virtual memory?",
        "explicit_doc_refs": [],
        "allowed_documents": None,  # Any document allowed
        "expected_behavior": "Global retrieval over indexed corpus."
    },
    {
        "test_id": "test_2_explicit_doc_missing_info",
        "query": "According to OS_Notes.txt, explain paging.",
        "explicit_doc_refs": ["OS_Notes.txt"],
        "allowed_documents": ["OS_Notes.txt"],
        "expected_behavior": "Strict OS_Notes.txt scope, refusal because paging is absent."
    },
    {
        "test_id": "test_3_explicit_doc_available_info",
        "query": "According to OS_Notes.txt, what is virtual memory?",
        "explicit_doc_refs": ["OS_Notes.txt"],
        "allowed_documents": ["OS_Notes.txt"],
        "expected_behavior": "Strict OS_Notes.txt scope, grounded answer returned."
    },
    {
        "test_id": "test_4_explicit_paging_doc",
        "query": "According to sample_routing_doc.txt, explain paging.",
        "explicit_doc_refs": ["sample_routing_doc.txt"],
        "allowed_documents": ["sample_routing_doc.txt"],
        "expected_behavior": "Strict sample_routing_doc.txt scope, grounded paging answer returned."
    },
    {
        "test_id": "test_5_multi_doc_comparison",
        "query": "Compare the discussion of memory management in OS_Notes.txt and sample_exam.txt.",
        "explicit_doc_refs": ["OS_Notes.txt", "sample_exam.txt"],
        "allowed_documents": ["OS_Notes.txt", "sample_exam.txt"],
        "expected_behavior": "Strict OS_Notes.txt & sample_exam.txt scope, no external docs allowed."
    },
    {
        "test_id": "test_6_no_doc_ref_paging",
        "query": "Explain paging.",
        "explicit_doc_refs": [],
        "allowed_documents": None,
        "expected_behavior": "Global retrieval, paging answer delivered."
    }
]

regression_results = []
all_tests_passed = True

for tdef in test_definitions:
    tid = tdef["test_id"]
    qtext = tdef["query"]
    allowed_docs = tdef["allowed_documents"]

    print(f"\n------------------------------------------------------------------")
    print(f"RUNNING REGRESSION TEST [{tid}]: '{qtext}'")
    print(f"------------------------------------------------------------------")

    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/query", json={"question": qtext}, timeout=120)
    t1 = time.time()
    wall_lat = round(t1 - t0, 4)

    status_code = resp.status_code
    resp_data = resp.json() if status_code == 200 else {}

    answer = resp_data.get("answer", "")
    context = resp_data.get("context", [])

    # Extract source filenames from context chunks
    retrieved_docs = []
    for c in context:
        m = re.search(r"\[Document: (.*?) \|", c)
        if m:
            retrieved_docs.append(m.group(1))

    unique_retrieved_docs = list(set(retrieved_docs))

    # Evaluate Scope Enforcement Invariant: FINAL_CONTEXT.documents ⊆ REQUESTED_DOCUMENTS
    forbidden_docs = []
    scope_respected = True

    if allowed_docs is not None:
        for d in unique_retrieved_docs:
            if d not in allowed_docs:
                forbidden_docs.append(d)
                scope_respected = False

    # Poll LangSmith for completed trace
    ls_run_data = None
    for attempt in range(1, 8):
        time.sleep(1)
        runs = list(ls_client.list_runs(project_id=ls_project_id, is_root=True, limit=10))
        for r in runs:
            if (r.inputs.get("question") == qtext or r.inputs.get("query") == qtext) and r.outputs is not None:
                ls_run_data = {
                    "run_id": str(r.id),
                    "completed": r.outputs is not None and r.end_time is not None,
                    "latency_sec": round(r.end_time.timestamp() - r.start_time.timestamp(), 4) if r.end_time else 0,
                    "has_inputs": r.inputs is not None,
                    "has_outputs": r.outputs is not None
                }
                break
        if ls_run_data:
            break

    if not ls_run_data:
        ls_run_data = {
            "run_id": "NOT_FOUND",
            "completed": False,
            "latency_sec": 0,
            "has_inputs": False,
            "has_outputs": False
        }

    # Evaluate overall test result
    test_passed = status_code == 200 and scope_respected
    if tid == "test_2_explicit_doc_missing_info" and "I cannot find" not in answer:
        test_passed = False

    if not test_passed:
        all_tests_passed = False

    result_str = "PASS" if test_passed else "FAIL"

    print(f"  HTTP Status Code   : {status_code}")
    print(f"  HTTP Wall Latency  : {wall_lat}s")
    print(f"  Requested Scope    : {allowed_docs if allowed_docs else 'ALL_DOCUMENTS'}")
    print(f"  Retrieved Documents: {unique_retrieved_docs}")
    print(f"  Forbidden Documents: {forbidden_docs}")
    print(f"  Scope Respected    : {scope_respected}")
    print(f"  Answer Snippet     : {repr(answer[:120])}...")
    print(f"  Test Result        : {result_str}")

    rec = {
        "test_id": tid,
        "query": qtext,
        "document_reference": tdef["explicit_doc_refs"],
        "allowed_documents": allowed_docs if allowed_docs else ["*"],
        "retrieved_documents": unique_retrieved_docs,
        "forbidden_documents": forbidden_docs,
        "scope_respected": scope_respected,
        "context_count": len(context),
        "answer": answer,
        "http_status": status_code,
        "http_latency": wall_lat,
        "langsmith": ls_run_data,
        "result": result_str
    }
    regression_results.append(rec)

summary = {
    "phase": "5A.11",
    "status": "PASS" if all_tests_passed else "FAIL",
    "total_tests": len(regression_results),
    "passed_tests": sum(1 for r in regression_results if r["result"] == "PASS"),
    "failed_tests": sum(1 for r in regression_results if r["result"] == "FAIL"),
    "tests": regression_results
}

os.makedirs("evaluation/results", exist_ok=True)
with open("evaluation/results/phase5a11_document_scope_regression.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\n==================================================================")
print(f"REGRESSION SUITE COMPLETE: Status = {summary['status']} ({summary['passed_tests']}/{summary['total_tests']} Passed)")
print(f"Saved results to evaluation/results/phase5a11_document_scope_regression.json")
print(f"==================================================================")
