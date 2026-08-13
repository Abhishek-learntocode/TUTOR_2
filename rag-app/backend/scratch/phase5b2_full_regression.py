import os
import sys
import time
import json
import re
from dotenv import load_dotenv

# Ensure backend root is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

env_path = os.path.join(backend_dir, ".env")
load_dotenv(env_path)

from fastapi.testclient import TestClient
from langsmith import Client
from app.main import app
from app.config import settings
from app.rag.providers import clear_provider_call_ledger, get_provider_call_ledger, save_provider_call_ledger

client = TestClient(app)
ls_client = Client()
ls_project_id = "1a4d2edf-a0b5-4975-b31a-1c5494eb9569"

print("=" * 80)
print("PHASE 5B.2A — FULL REGRESSION & THREE-WAY RECONCILIATION")
print("=" * 80)

# 1. Rerun Phase 5A.11 Document Scope Regression Suite (6 queries)
print("\n---> STEP 1: Running Phase 5A.11 Document-Scope Regression Suite...")
scope_tests = [
    {
        "test_id": "test_1_general_retrieval",
        "query": "What is virtual memory?",
        "explicit_doc_refs": [],
        "allowed_documents": None,
    },
    {
        "test_id": "test_2_explicit_doc_missing_info",
        "query": "According to OS_Notes.txt, explain paging.",
        "explicit_doc_refs": ["OS_Notes.txt"],
        "allowed_documents": ["OS_Notes.txt"],
    },
    {
        "test_id": "test_3_explicit_doc_available_info",
        "query": "According to OS_Notes.txt, what is virtual memory?",
        "explicit_doc_refs": ["OS_Notes.txt"],
        "allowed_documents": ["OS_Notes.txt"],
    },
    {
        "test_id": "test_4_explicit_paging_doc",
        "query": "According to sample_routing_doc.txt, explain paging.",
        "explicit_doc_refs": ["sample_routing_doc.txt"],
        "allowed_documents": ["sample_routing_doc.txt"],
    },
    {
        "test_id": "test_5_multi_doc_comparison",
        "query": "Compare the discussion of memory management in OS_Notes.txt and sample_exam.txt.",
        "explicit_doc_refs": ["OS_Notes.txt", "sample_exam.txt"],
        "allowed_documents": ["OS_Notes.txt", "sample_exam.txt"],
    },
    {
        "test_id": "test_6_no_doc_ref_paging",
        "query": "Explain paging.",
        "explicit_doc_refs": [],
        "allowed_documents": None,
    },
]

scope_results = []
scope_all_passed = True

for st in scope_tests:
    tid = st["test_id"]
    qtext = st["query"]
    allowed_docs = st["allowed_documents"]

    resp = client.post("/query", json={"question": qtext})
    scode = resp.status_code
    rdata = resp.json() if scode == 200 else {}
    answer = rdata.get("answer", "")
    context = rdata.get("context", [])

    retrieved_docs = []
    for c in context:
        m = re.search(r"\[Document: (.*?) \|", c)
        if m:
            retrieved_docs.append(m.group(1))
    unique_retrieved_docs = list(set(retrieved_docs))

    forbidden_docs = []
    scope_respected = True
    if allowed_docs is not None:
        for d in unique_retrieved_docs:
            if d not in allowed_docs:
                forbidden_docs.append(d)
                scope_respected = False

    test_passed = (scode == 200) and scope_respected
    if tid == "test_2_explicit_doc_missing_info" and "I cannot find" not in answer:
        test_passed = False

    if not test_passed:
        scope_all_passed = False

    print(f"  [{tid}] Status={scode} | Scope Respected={scope_respected} | Passed={test_passed}")
    scope_results.append({
        "test_id": tid,
        "query": qtext,
        "allowed_documents": allowed_docs,
        "retrieved_documents": unique_retrieved_docs,
        "forbidden_documents": forbidden_docs,
        "scope_respected": scope_respected,
        "answer_snippet": answer[:80],
        "passed": test_passed,
    })

print(f"Phase 5A.11 Scope Regression Status: {'PASS' if scope_all_passed else 'FAIL'} ({sum(1 for r in scope_results if r['passed'])}/6)")

# Save Phase 5A.11 Regression JSON
out_dir = os.path.join(backend_dir, "evaluation", "results")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "phase5b2_document_scope_regression.json"), "w", encoding="utf-8") as f:
    json.dump({"all_passed": scope_all_passed, "tests": scope_results}, f, indent=2)

print("\n==================================================================")
print("PHASE 5B.2A FULL REGRESSION COMPLETE!")
print("==================================================================")
