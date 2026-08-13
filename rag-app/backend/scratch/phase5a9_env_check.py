import os
import sys
import time
import requests
import json

sys.path.insert(0, os.path.abspath("."))

from app.config import settings
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore

print("=== PHASE 5A.9 — STEP 1: ENVIRONMENT & VECTOR STORE VALIDATION ===")

# 1. Backend Health
try:
    r_health = requests.get("http://127.0.0.1:8000/health", timeout=5)
    print(f"1. Backend GET /health: Status {r_health.status_code} | Body: {r_health.json()}")
except Exception as e:
    print(f"1. Backend GET /health FAILED: {e}")

# 2. Ollama Version
try:
    r_ver = requests.get("http://localhost:11434/api/version", timeout=5)
    print(f"2. Ollama GET /api/version: Status {r_ver.status_code} | Body: {r_ver.json()}")
except Exception as e:
    print(f"2. Ollama GET /api/version FAILED: {e}")

# 3. Ollama Models
try:
    r_tags = requests.get("http://localhost:11434/api/tags", timeout=5)
    models = [m['name'] for m in r_tags.json().get('models', [])]
    print(f"3. Ollama Available Models: {models}")
except Exception as e:
    print(f"3. Ollama GET /api/tags FAILED: {e}")

# 4. Vector Store Inspection (Read-Only)
emb_svc = EmbeddingService(
    provider=settings.embedding_provider,
    model_name=settings.embedding_model,
    base_url=settings.llm_base_url,
)
vs = VectorStore(embeddings=emb_svc.embeddings, store_path=settings.vector_store_path)
all_docs = vs.get_all_documents()

chunks_per_file = {}
for d in all_docs:
    fn = d.metadata.get("source_filename", "UNKNOWN")
    chunks_per_file.setdefault(fn, []).append(d)

print(f"\n4. Vector Store Path: {settings.vector_store_path}")
print(f"   - Total Vector Count: {len(all_docs)}")
print(f"   - FAISS Dimension: {vs.store.index.d if vs.store else 'None'}")
print(f"   - Unique Source Filenames Count: {len(chunks_per_file)}")
print("   - Breakdown per Source File:")
for fn, chunks in sorted(chunks_per_file.items()):
    print(f"     * '{fn}': {len(chunks)} chunk(s)")

env_summary = {
    "backend_health": r_health.json() if r_health.status_code == 200 else None,
    "ollama_version": r_ver.json() if r_ver.status_code == 200 else None,
    "ollama_models": models if r_tags.status_code == 200 else [],
    "vector_store": {
        "total_chunks": len(all_docs),
        "dimension": vs.store.index.d if vs.store else None,
        "chunks_per_file": {fn: len(c) for fn, c in sorted(chunks_per_file.items())}
    }
}

os.makedirs("scratch", exist_ok=True)
with open("scratch/phase5a9_env_summary.json", "w", encoding="utf-8") as f:
    json.dump(env_summary, f, indent=2)

print("\nEnvironment validation complete. Saved summary to scratch/phase5a9_env_summary.json.")
