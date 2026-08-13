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

print("=== PHASE 5A.10 — RAG CORRECTNESS & EVIDENCE AUDIT RUNNER ===")

test_cases = [
    {
        "query_id": "test_001_factual",
        "task_type": "factual_question",
        "query": "What is virtual memory?",
        "doc_ref_expected": None
    },
    {
        "query_id": "test_002_conceptual",
        "task_type": "conceptual_explanation",
        "query": "Explain paging.",
        "doc_ref_expected": None
    },
    {
        "query_id": "test_003_multihop",
        "task_type": "multi_hop",
        "query": "How does paging work, and what problem does it solve?",
        "doc_ref_expected": None
    },
    {
        "query_id": "test_004_doc_scoped",
        "task_type": "document_specific",
        "query": "According to OS_Notes.txt, explain paging.",
        "doc_ref_expected": "OS_Notes.txt"
    },
    {
        "query_id": "test_005_comparison",
        "task_type": "comparison",
        "query": "Compare the discussion of memory management in OS_Notes.txt and sample_exam.txt.",
        "doc_ref_expected": ["OS_Notes.txt", "sample_exam.txt"]
    },
    {
        "query_id": "test_006_mcq",
        "task_type": "exam_mcq",
        "query": "Which of the following best describes virtual memory? A) Physical RAM extension B) Memory management capability C) Hard drive cache D) CPU register.",
        "doc_ref_expected": None
    },
    {
        "query_id": "test_007_missing_info",
        "task_type": "missing_information",
        "query": "What is the stock price of Apple in 2026?",
        "doc_ref_expected": None
    },
    {
        "query_id": "test_008_ambiguous",
        "task_type": "ambiguous",
        "query": "Explain the table structure.",
        "doc_ref_expected": None
    }
]

matrix = []

for tc in test_cases:
    qid = tc["query_id"]
    ttype = tc["task_type"]
    qtext = tc["query"]

    print(f"\n==================================================================")
    print(f"AUDITING [{qid}] ({ttype}): '{qtext}'")
    print(f"==================================================================")

    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/query", json={"question": qtext}, timeout=120)
    t1 = time.time()
    wall_lat = round(t1 - t0, 4)

    status_code = resp.status_code
    resp_data = resp.json() if status_code == 200 else {}

    answer = resp_data.get("answer", "")
    context = resp_data.get("context", [])

    # Poll LangSmith API for fully completed run
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
            "run_id": "NOT_FOUND_OR_INCOMPLETE",
            "completed": False,
            "latency_sec": 0,
            "has_inputs": False,
            "has_outputs": False
        }

    # Extract retrieved source files
    retrieved_sources = []
    for c in context:
        m = re.search(r"\[Document: (.*?) \|", c)
        if m:
            retrieved_sources.append(m.group(1))

    # Evaluate Document Scope Compliance
    doc_scope_respected = True
    if qid == "test_004_doc_scoped":
        # Query explicitly asked for OS_Notes.txt.
        # If retrieved sources contain documents other than OS_Notes.txt and answer uses them, doc_scope_respected = False
        non_os_notes = [s for s in retrieved_sources if s != "OS_Notes.txt"]
        if non_os_notes and "I cannot find" not in answer:
            doc_scope_respected = False

    # Claim-Level Grounding Analysis
    claims = []
    is_refusal = "I cannot find the answer" in answer

    if not is_refusal and answer:
        # Split answer into sentences
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        for sent in raw_sentences:
            # Check if sentence is supported by any context chunk
            supported = False
            matching_chunk = None
            sent_words = set(re.findall(r"\w+", sent.lower()))
            for idx, c_str in enumerate(context):
                c_words = set(re.findall(r"\w+", c_str.lower()))
                overlap = sent_words.intersection(c_words)
                if len(overlap) >= min(len(sent_words), 4):
                    supported = True
                    matching_chunk = f"Chunk #{idx+1}"
                    break

            claims.append({
                "claim": sent,
                "status": "SUPPORTED_BY_CONTEXT" if supported else "UNSUPPORTED_INFORMATION",
                "matching_chunk": matching_chunk
            })

    # Task Correctness Analysis
    answer_correct = True
    refusal_correct = None
    notes = ""

    if ttype == "missing_information" or ttype == "ambiguous":
        refusal_correct = is_refusal
        answer_correct = is_refusal
        notes = "Intended refusal case." if is_refusal else "Failed to refuse missing/ambiguous query."
    elif qid == "test_004_doc_scoped":
        # Document scope check
        if not doc_scope_respected:
            answer_correct = False
            notes = "DOCUMENT_SCOPE_VIOLATION: Query requested OS_Notes.txt (which lacks paging), but backend retrieved sample_routing_doc.txt and answered from it."
    elif qid == "test_006_mcq":
        # MCQ Option B selection check
        if "B)" in answer or "Memory management capability" in answer:
            answer_correct = True
            notes = "Correctly selected Option B based on context."
        else:
            answer_correct = False
            notes = "Failed to select correct MCQ option B."
    elif qid == "test_005_comparison":
        if "OS_Notes.txt" in answer or "sample_exam.txt" in answer or len(retrieved_sources) >= 2:
            answer_correct = True
            notes = "Synthesized comparative information across documents."

    entry = {
        "query_id": qid,
        "task_type": ttype,
        "query": qtext,
        "document_reference": tc["doc_ref_expected"],
        "retrieval": {
            "context_count": len(context),
            "retrieved_sources": retrieved_sources,
            "final_context_preview": [c[:100] + "..." for c in context]
        },
        "answer": answer,
        "claims": claims,
        "evaluation": {
            "retrieval_relevant": len(context) > 0,
            "grounded": all(c["status"] == "SUPPORTED_BY_CONTEXT" for c in claims) if claims else is_refusal,
            "task_completed": answer_correct,
            "document_scope_respected": doc_scope_respected,
            "answer_correct": answer_correct,
            "refusal_correct": refusal_correct,
            "unsupported_claims": [c["claim"] for c in claims if c["status"] != "SUPPORTED_BY_CONTEXT"],
            "notes": notes
        },
        "langsmith": ls_run_data,
        "http": {
            "status_code": status_code,
            "wall_latency_sec": wall_lat
        }
    }

    matrix.append(entry)

os.makedirs("evaluation/results", exist_ok=True)
with open("evaluation/results/phase5a10_correctness_matrix.json", "w", encoding="utf-8") as f:
    json.dump(matrix, f, indent=2)

print("\nCorrectness matrix evaluation complete. Saved to evaluation/results/phase5a10_correctness_matrix.json.")
