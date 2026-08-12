# CLAUDE.md — LexRAG: Indian Legal Document Intelligence RAG

## Project Identity
**Project Name:** LexRAG — Agentic Legal Intelligence System  
**Resume Title:** Production Agentic RAG System for Indian Legal Document Analysis  
**Base Repository:** https://github.com/d-hackmt/8hr-MARATHON  
**Your Repository:** Fork → rename to `lexrag` or `legal-rag-india`

---

## Why This Base
The 8hr-MARATHON repo has a genuinely production-grade stack that would take weeks
to build from scratch: NeMo Guardrails, Portkey LLM Gateway with fallback, FlashRank
local reranker, RAGAS 6-metric eval suite, Pydantic Logfire + LangSmith dual
observability, Dockerfile, .gcloudignore. The infrastructure is real.

The ONLY reason it is not resume-ready as-is: 104 forks exist, "True Data / Noisy Data"
framing is meaningless and every forker kept it, the domain is generic. Your job is to
make the DOMAIN completely your own while keeping the production infrastructure.

---

## What This Project Becomes
An agentic RAG system that lets lawyers, researchers, and compliance officers query
Indian legal documents — eCourts judgments, SEBI circulars, RBI monetary policy
documents, and Companies Act sections — in natural language. The agent routes
conversational queries (explaining legal terms) vs technical queries (finding specific
precedents) via a LangGraph Planner node, retrieves and reranks relevant legal
passages, enforces domain guardrails (no legal advice given, no hallucinated citations),
and produces grounded answers with section references.

---

## Tech Stack (keep from original, nothing removed)
| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph (Planner → Retriever → Responder nodes) |
| LLM | Groq Llama 3.3 70B via Portkey Gateway (primary), Llama 3.1 8B (fallback) |
| Guardrails | NeMo Guardrails — domain-specific COLANG config |
| Vector DB | Qdrant Cloud |
| Reranker | FlashRank (local, zero-latency cross-encoder) |
| Embeddings | Gemini `gemini-embedding-2-preview` (3072-dim) |
| Evaluation | RAGAS (6 metrics) + custom Jaccard Tool Correctness |
| Observability | Pydantic Logfire + LangSmith |
| API | FastAPI `/query` endpoint |
| UI | Streamlit chat + Streamlit eval app |
| Ingestion | Local parsers: pypdf, beautifulsoup4, pdfplumber, python-docx |

---

## Folder Structure (what changes vs what stays)

```
lexrag/
├── app/
│   ├── agents/
│   │   └── nodes/
│   │       ├── planner_node.py      ← MODIFY: update system prompt for legal domain
│   │       ├── retriever_node.py    ← MODIFY: add BM25 hybrid search + RRF fusion
│   │       └── responder_node.py    ← MODIFY: update prompt to cite section numbers
│   ├── gateway/                     ← KEEP AS-IS
│   ├── guardrails/
│   │   └── config/
│   │       ├── config.yml           ← REPLACE: legal-domain COLANG flows
│   │       └── legal_rails.co      ← CREATE: new domain-specific rail definitions
│   ├── ingestion/
│   │   ├── chunking/                ← MODIFY: reduce chunk size to 800 chars for legal
│   │   └── loaders/                 ← KEEP AS-IS (pypdf handles legal PDFs fine)
│   ├── services/
│   │   └── retrieval/
│   │       ├── vector_search.py     ← MODIFY: add BM25 alongside Qdrant
│   │       └── hybrid_fusion.py     ← CREATE: RRF fusion function
│   ├── config.py                    ← MODIFY: update collection name, domain constants
│   └── main.py                      ← KEEP AS-IS
├── evals/
│   ├── golden_dataset.json          ← REPLACE: 50 legal Q&A pairs (see spec below)
│   └── app.py                       ← KEEP AS-IS
├── ui/
│   └── app.py                       ← MODIFY: update title, placeholder text, examples
├── DATA/                            ← REPLACE ENTIRELY (see data sources below)
├── DOCS/                            ← UPDATE: docs 01 and 03 need domain updates
├── requirements.txt                 ← ADD: rank_bm25
├── .env.example                     ← KEEP AS-IS
├── Dockerfile                       ← KEEP AS-IS
└── README.md                        ← REWRITE: your project, your domain, your numbers
```

---

## Priority 1 — Data Sources (Day 1, Critical)

### What data to use
All sources are free and publicly available:

1. **SEBI Circulars** — https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doListing=yes&sid=5&ssid=11&smid=0  
   Download last 3 years of circulars as PDFs. ~200-300 documents.

2. **RBI Master Directions** — https://www.rbi.org.in/scripts/BS_ViewMasDirections.aspx  
   Master Directions on KYC, NBFC, Payment Systems. ~50 documents.

3. **Companies Act 2013 Sections** — Download from MCA website as PDF.  
   https://www.mca.gov.in/content/mca/global/en/acts-rules/ebooks/acts.html

4. **eCourts Judgments (optional)** — https://indiankanoon.org/  
   You can scrape or use their API for landmark Supreme Court judgments.
   Start with 50-100 well-known judgments if using this source.

### Ingestion command (unchanged from original)
```bash
python -m app.ingestion.processor DATA --wipe
```

### Data volume target
Aim for 500-1000 documents total. This gives you enough density that the retrieval
story is credible. Do not fake this number — run it and report the actual count.

---

## Priority 2 — Domain-Specific NeMo Guardrails (Day 1, Critical)

### Why this matters for your resume
The original guardrails are generic jailbreak/off-topic filters. Any forker kept
them. Your guardrails should be domain-tuned, which shows you actually understood
the NeMo architecture, not just cloned the repo.

### Replace `app/guardrails/config/config.yml` with:
```yaml
models:
  - type: main
    engine: groq
    model: llama-3.3-70b-versatile

rails:
  input:
    flows:
      - legal domain check
      - no legal advice disclaimer
      - citation hallucination block
  output:
    flows:
      - response must cite source
      - no definitive legal advice
```

### Create `app/guardrails/config/legal_rails.co`:
```colang
define user ask legal advice
  "am I liable"
  "will I win this case"
  "is this illegal"
  "what should I do legally"
  "give me legal advice"

define bot refuse legal advice
  "I can explain what the law says but cannot provide legal advice.
   Please consult a qualified advocate for case-specific guidance."

define flow no legal advice disclaimer
  user ask legal advice
  bot refuse legal advice

define user ask off topic
  "what is the weather"
  "tell me a joke"
  "write me code"

define bot redirect to legal
  "I am specialized for Indian legal document analysis.
   Please ask me about SEBI regulations, RBI directives, or Companies Act provisions."

define flow legal domain check
  user ask off topic
  bot redirect to legal
```

---

## Priority 3 — BM25 Hybrid Search (Day 2, Important)

### Create `app/services/retrieval/hybrid_fusion.py`
```python
from rank_bm25 import BM25Okapi
from typing import List, Tuple
import numpy as np

def reciprocal_rank_fusion(
    semantic_results: List[Tuple[str, float]],
    bm25_results: List[Tuple[str, float]],
    k: int = 60,
    semantic_weight: float = 0.7,
    bm25_weight: float = 0.3
) -> List[Tuple[str, float]]:
    """
    RRF fusion of semantic (Qdrant) and BM25 results.
    k=60 is the standard constant from the original RRF paper.
    semantic_weight=0.7 because embedding models capture legal semantics better
    than BM25 for long regulatory text, but BM25 helps with exact section citations.
    """
    scores = {}

    for rank, (doc_id, _) in enumerate(semantic_results):
        scores[doc_id] = scores.get(doc_id, 0) + semantic_weight * (1 / (k + rank + 1))

    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + bm25_weight * (1 / (k + rank + 1))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### Modify `app/services/retrieval/vector_search.py`
Add BM25 index construction at ingestion time, store corpus in memory or Redis,
run BM25 alongside Qdrant on every query, fuse with RRF before passing to FlashRank.

Add to requirements.txt:
```
rank_bm25
```

---

## Priority 4 — RAGAS Golden Dataset (Day 2, Critical)

### Replace `evals/golden_dataset.json`
Write 50 real Q&A pairs. Structure:
```json
[
  {
    "question": "What is the timeline for filing a complaint under SEBI (LODR) Regulations 2015?",
    "ground_truth": "Under SEBI (LODR) Regulations 2015, listed entities must file complaints within 3 working days of receipt...",
    "context_document": "sebi_lodr_2015.pdf",
    "category": "procedural"
  },
  {
    "question": "Define 'related party transaction' as per Companies Act 2013 Section 2(76)",
    "ground_truth": "Section 2(76) of the Companies Act 2013 defines related party as...",
    "context_document": "companies_act_2013.pdf",
    "category": "definition"
  }
]
```

Aim for 5 categories: definitions, procedural, penalty/enforcement, eligibility, comparative.
10 questions per category = 50 total. Source answers directly from the documents you ingested.
Do NOT generate these with GPT — write them manually from the actual documents.
This is what makes them genuine and defensible.

---

## Priority 5 — Node Prompt Updates (Day 2)

### `app/agents/nodes/planner_node.py` — update system prompt
```python
PLANNER_SYSTEM_PROMPT = """You are a legal research assistant specialized in Indian
corporate and securities law.

Classify the user's query as one of:
- "technical": requires searching specific legal provisions, sections, case law, or
  regulatory circulars (route to Retriever)
- "conversational": general explanation of legal concepts, definitions, or
  clarifications (route to Responder directly)

You MUST NOT provide legal advice or predict legal outcomes.
You MUST cite specific sections, circular numbers, or judgment names in responses."""
```

### `app/agents/nodes/responder_node.py` — update system prompt
```python
RESPONDER_SYSTEM_PROMPT = """You are LexRAG, a legal research assistant for Indian
corporate and securities law.

ALWAYS:
- Cite the specific section, circular number, or judgment you are drawing from
- Use the phrase 'According to [source]' before each factual claim
- Add the disclaimer: 'This is a research summary, not legal advice.'

NEVER:
- Predict legal outcomes for specific cases
- Say 'you should' or 'you must' in an advisory tone
- Generate citation numbers you did not retrieve from the provided context"""
```

---

## Priority 6 — UI Updates (Day 3, Quick)

### `ui/app.py` — update these strings only
```python
APP_TITLE = "LexRAG — Indian Legal Document Intelligence"
APP_SUBTITLE = "Query SEBI circulars, RBI directives, and Companies Act provisions"

EXAMPLE_QUERIES = [
    "What are the disclosure requirements for insider trading under SEBI regulations?",
    "Explain Section 135 of the Companies Act 2013 on CSR obligations",
    "What penalties apply for late filing of financial results under LODR?",
    "Define 'significant beneficial owner' under Companies Act",
]

DISCLAIMER = """⚖️ LexRAG is a research tool. Responses are not legal advice.
Consult a qualified advocate for case-specific guidance."""
```

---

## Priority 7 — README Rewrite (Day 3)

The README must contain:
1. Problem statement (why Indian legal research is hard)
2. Architecture diagram (already exists in ARCHITECTURE.md — copy it)
3. Data sources (SEBI, RBI, Companies Act — link to official sources)
4. **Before/after RAGAS numbers table** (run the eval suite, copy results)
5. How to run (keep the same commands, update collection name)
6. Sample queries with example outputs

### The numbers table template (fill in after running evals):
```markdown
## Evaluation Results (RAGAS)

| Metric | Score |
|---|---|
| Faithfulness | X.XX |
| Answer Relevancy | X.XX |
| Context Precision | X.XX |
| Context Recall | X.XX |
| Answer Correctness | X.XX |
| Tool Correctness (Jaccard) | X.XX |

Evaluated on 50 legal Q&A pairs across SEBI, RBI, and Companies Act documents.
```

---

## What NOT to Change
- `app/gateway/` — Portkey routing code is solid, leave it
- `app/ingestion/loaders/` — all parsers work for PDFs, leave them
- `evals/app.py` — the Streamlit eval UI logic is fine
- `Dockerfile` — works as-is
- `.gcloudignore` — works as-is
- `requirements.txt` — only ADD `rank_bm25`, do not remove anything

---

## Final Resume Bullet
```
LexRAG: Agentic RAG system over Indian legal corpus (SEBI/RBI/Companies Act).
LangGraph Planner→Retriever→Responder with NeMo domain guardrails, Portkey LLM
gateway (Llama 3.3 70B), hybrid BM25+Qdrant retrieval with RRF fusion, FlashRank
reranker, RAGAS eval (Faithfulness: X.XX, Relevancy: X.XX). Logfire + LangSmith
observability. Dockerized, GCP-ready.
```

---

## Interview Questions You Must Prepare

**On the agent architecture:**
- Walk me through what happens from user query to response (know every node)
- Why LangGraph over a simple LangChain chain?
- How does MemorySaver work across conversation turns?
- What is a COLANG flow and how does NeMo intercept the request?

**On retrieval:**
- Why did you add BM25 alongside Qdrant?
- What is RRF and what is the k=60 constant?
- What does FlashRank do that Qdrant reranking doesn't?
- Why Gemini embeddings at 3072 dimensions for legal text?

**On evaluation:**
- What does Faithfulness measure specifically?
- How is Context Precision different from Context Recall?
- Why did you build the golden dataset manually?
- What does your Jaccard Tool Correctness metric check?

**On guardrails:**
- What is a COLANG flow?
- How does NeMo intercept before the LLM call?
- Why specifically block "legal advice" queries?

**Domain question (always asked):**
- Why Indian legal documents specifically?
- What makes legal text harder to retrieve than general text?
  (Answer: long documents, dense section cross-references, specific terminology
  that BM25 handles better than semantic search for exact statute citations)
