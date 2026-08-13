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

from langsmith import Client
from app.config import settings
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore
from app.rag.bm25_retriever import BM25Retriever
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.providers import get_provider
from app.rag.llm import LLMService
from app.rag.query_analyzer import QueryAnalyzer
from app.graph.nodes import RAGNodes
from app.graph.workflow import RAGGraph
from app.models.state import RAGState

ls_client = Client()
ls_project_id = "1a4d2edf-a0b5-4975-b31a-1c5494eb9569"

print("=" * 80)
print("PHASE 5B.0 — MULTI-MODEL PROVIDER ARCHITECTURE & MODEL BENCHMARK")
print("=" * 80)

# Load evaluation dataset (35 queries)
dataset_path = os.path.join(backend_dir, "evaluation", "datasets", "rag_baseline_v1.jsonl")
eval_dataset = []
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            eval_dataset.append(json.loads(line.strip()))

print(f"[*] Loaded {len(eval_dataset)} evaluation queries from {dataset_path}")

# Initialize shared retrieval components once
embedding_service = EmbeddingService(
    provider=settings.embedding_provider,
    model_name=settings.embedding_model,
    base_url=settings.llm_base_url,
)
vector_store = VectorStore(
    embeddings=embedding_service.embeddings,
    store_path=settings.vector_store_path,
)
bm25_retriever = BM25Retriever()
reranker = Reranker(model_name=settings.reranker_model)
retriever = Retriever(
    vector_store=vector_store,
    bm25_retriever=bm25_retriever,
    reranker=reranker,
    top_k_candidates=settings.top_k_candidates,
    top_k_final=settings.top_k_final,
)

# Benchmark Candidate Matrix
CONFIGS = {
    "CONFIG_A_LOCAL_BASELINE": {
        "description": "Local Baseline (Ollama Qwen2.5 1.5B for QA + AG)",
        "qa_provider": "ollama",
        "qa_model": "qwen2.5:1.5b",
        "ag_provider": "ollama",
        "ag_model": "qwen2.5:1.5b",
    },
    "CONFIG_B_OPENROUTER_FAST_STRONG": {
        "description": "OpenRouter Role Specialization (Fast QA + Strong AG)",
        "qa_provider": "openrouter",
        "qa_model": "google/gemma-4-26b-a4b-it:free",
        "ag_provider": "openrouter",
        "ag_model": "google/gemma-4-31b-it:free",
    },
    "CONFIG_C_LOCAL_QA_OPENROUTER_STRONG": {
        "description": "Hybrid Local QA + Strong OpenRouter AG",
        "qa_provider": "ollama",
        "qa_model": "qwen2.5:1.5b",
        "ag_provider": "openrouter",
        "ag_model": "google/gemma-4-31b-it:free",
    },
    "CONFIG_D_OPENROUTER_FAST_LOCAL_AG": {
        "description": "Hybrid Fast OpenRouter QA + Local Ollama AG",
        "qa_provider": "openrouter",
        "qa_model": "google/gemma-4-26b-a4b-it:free",
        "ag_provider": "ollama",
        "ag_model": "qwen2.5:1.5b",
    },
    "CONFIG_E_OPENROUTER_FREE_ROUTER": {
        "description": "OpenRouter Free Router for QA + AG",
        "qa_provider": "openrouter",
        "qa_model": "openrouter/free",
        "ag_provider": "openrouter",
        "ag_model": "openrouter/free",
    },
}

all_benchmark_results = {}

for cfg_key, cfg in CONFIGS.items():
    print(f"\n==================================================================")
    print(f"BENCHMARKING CONFIGURATION: {cfg_key}")
    print(f"Description: {cfg['description']}")
    print(f"QA: {cfg['qa_provider']} / {cfg['qa_model']}")
    print(f"AG: {cfg['ag_provider']} / {cfg['ag_model']}")
    print(f"==================================================================")

    # Build RAG pipeline graph for this config
    qa_p = get_provider(
        provider_name=cfg["qa_provider"],
        model_name=cfg["qa_model"],
        base_url=settings.llm_base_url,
        api_key=settings.openrouter_api_key,
    )
    ag_p = get_provider(
        provider_name=cfg["ag_provider"],
        model_name=cfg["ag_model"],
        base_url=settings.llm_base_url,
        api_key=settings.openrouter_api_key,
    )

    qa = QueryAnalyzer(provider=qa_p)
    llm_svc = LLMService(provider=ag_p)

    nodes = RAGNodes(retriever=retriever, llm=llm_svc, query_analyzer=qa)
    graph = RAGGraph(nodes=nodes)
    compiled_graph = graph.compile()

    config_results = []

    for item in eval_dataset:
        qid = item["id"]
        category = item["category"]
        qtext = item["query"]
        expected_behavior = item["expected_behavior"]
        doc_scope = item.get("document_scope")

        print(f"\n--- [{cfg_key}] QUERY {qid} ({category}): '{qtext[:60]}...' ---")

        start_t = time.time()
        error_msg = None
        final_state = None
        try:
            state_in = RAGState(question=qtext)
            final_state = compiled_graph.invoke(state_in)
            total_lat = round(time.time() - start_t, 4)
            status_code = 200
        except Exception as e:
            total_lat = round(time.time() - start_t, 4)
            status_code = 500
            error_msg = str(e)
            print(f"[ERROR] Invocation failed: {e}")

        answer = ""
        context = []
        q_type = "single_hop"
        sub_queries = []
        if final_state:
            answer = getattr(final_state, "answer", "") or ""
            context = getattr(final_state, "context", []) or []
            q_type = getattr(final_state, "query_type", "single_hop") or "single_hop"
            sub_queries = getattr(final_state, "sub_queries", []) or []

        # Extract retrieved document names
        retrieved_docs = []
        for c in context:
            m = re.search(r"\[Document: (.*?) \|", c)
            if m:
                retrieved_docs.append(m.group(1))
        unique_retrieved_docs = list(set(retrieved_docs))

        # Check scope enforcement
        scope_respected = True
        forbidden_docs = []
        if doc_scope is not None:
            for d in unique_retrieved_docs:
                if d not in doc_scope:
                    forbidden_docs.append(d)
                    scope_respected = False

        # Refusal check
        is_refusal = "I cannot find the answer in the provided context." in answer
        expected_refusal = expected_behavior == "refuse_or_state_not_in_context"
        refusal_correct = is_refusal if expected_refusal else (not is_refusal if context else True)

        # MCQ check
        mcq_correct = None
        if category == "exam_style" and "Which of the following" in qtext:
            # Option B is correct for Virtual memory question
            mcq_correct = "B" in answer or "Memory management" in answer or "option b" in answer.lower()

        # Poll LangSmith for completed root run
        ls_run_data = None
        for attempt in range(1, 8):
            time.sleep(1)
            try:
                runs = list(ls_client.list_runs(project_id=ls_project_id, is_root=True, limit=10))
                for r in runs:
                    if (r.inputs.get("question") == qtext or r.inputs.get("query") == qtext) and r.outputs is not None:
                        ls_run_data = {
                            "run_id": str(r.id),
                            "completed": r.outputs is not None and r.end_time is not None,
                            "latency_sec": round(r.end_time.timestamp() - r.start_time.timestamp(), 4) if r.end_time else 0,
                            "has_inputs": r.inputs is not None,
                            "has_outputs": r.outputs is not None,
                        }
                        break
            except Exception:
                pass
            if ls_run_data:
                break

        if not ls_run_data:
            ls_run_data = {
                "run_id": "NOT_FOUND",
                "completed": False,
                "latency_sec": 0,
                "has_inputs": False,
                "has_outputs": False,
            }

        rec = {
            "query_id": qid,
            "category": category,
            "query": qtext,
            "expected_behavior": expected_behavior,
            "document_scope": doc_scope,
            "query_analyzer_provider": cfg["qa_provider"],
            "query_analyzer_model": cfg["qa_model"],
            "answer_generator_provider": cfg["ag_provider"],
            "answer_generator_model": cfg["ag_model"],
            "actual_query_analyzer_model": getattr(qa_p, "model_name", cfg["qa_model"]),
            "actual_answer_generator_model": getattr(ag_p, "model_name", cfg["ag_model"]),
            "status_code": status_code,
            "total_latency": total_lat,
            "query_type": q_type,
            "sub_queries": sub_queries,
            "context_count": len(context),
            "retrieved_documents": unique_retrieved_docs,
            "forbidden_documents": forbidden_docs,
            "scope_respected": scope_respected,
            "answer": answer,
            "answer_length": len(answer),
            "is_refusal": is_refusal,
            "refusal_correctness": refusal_correct,
            "mcq_correctness": mcq_correct,
            "error": error_msg,
            "langsmith": ls_run_data,
            "three_way_reconciled": status_code == 200 and ls_run_data["completed"],
        }
        config_results.append(rec)

        print(f"  HTTP Status    : {status_code} | Total Latency: {total_lat}s")
        print(f"  Query Type     : {q_type} | Sub-queries: {len(sub_queries)}")
        print(f"  Context Chunks : {len(context)} | Docs: {unique_retrieved_docs}")
        print(f"  Answer Length  : {len(answer)} chars")
        print(f"  Answer Snippet : {repr(answer[:100])}...")

    all_benchmark_results[cfg_key] = {
        "config_info": cfg,
        "total_queries": len(config_results),
        "successful_queries": sum(1 for r in config_results if r["status_code"] == 200),
        "failed_queries": sum(1 for r in config_results if r["status_code"] != 200),
        "average_latency": round(sum(r["total_latency"] for r in config_results) / len(config_results), 4) if config_results else 0,
        "scope_pass_rate": round(sum(1 for r in config_results if r["scope_respected"]) / len(config_results), 4) if config_results else 0,
        "refusal_pass_rate": round(sum(1 for r in config_results if r["refusal_correctness"]) / len(config_results), 4) if config_results else 0,
        "queries": config_results,
    }

# Save output JSON
out_dir = os.path.join(backend_dir, "evaluation", "results")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "phase5b_model_benchmark.json")

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(all_benchmark_results, f, indent=2)

print("\n==================================================================")
print(f"[SUMMARY] Phase 5B.0 Benchmark Completed across {len(CONFIGS)} Configurations!")
print(f"[SUMMARY] Saved detailed benchmark data to {out_file}")
print("==================================================================")
