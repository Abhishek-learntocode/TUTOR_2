# Minimal RAG Application — Backend + Streamlit Frontend

A clean, minimal RAG (Retrieval-Augmented Generation) application built from scratch without unnecessary abstractions or complex multi-agent frameworks.

## Tech Stack & Architecture

- **Backend:** FastAPI
- **RAG Orchestration:** LangGraph (`StateGraph`)
- **Data Validation & State:** Pydantic (`RAGState`, `QueryRequest`, `QueryResponse`)
- **Vector Store:** FAISS (Local persistence)
- **Embeddings:** Ollama `nomic-embed-text:latest`
- **LLM:** Ollama `qwen2.5:1.5b`
- **Frontend:** Streamlit

---

## Installation & Setup

### 1. Prerequisites

Ensure local Ollama is running:

```bash
ollama pull nomic-embed-text:latest
ollama pull qwen2.5:1.5b
```

### 2. Install Dependencies

In your Python environment, install requirements:

```bash
pip install -r rag-app/backend/requirements.txt
pip install -r rag-app/frontend/requirements.txt
```

---

## Running the Application

### Start Backend API

Navigate to `rag-app/backend` and run:

```bash
cd rag-app/backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

*(Alternatively, if running directly from `rag-app/backend`, you can also run `python app/main.py` directly).*

The API will start at `http://127.0.0.1:8000`. You can test endpoints via Swagger docs at `http://127.0.0.1:8000/docs`.

### Start Streamlit Frontend

In another terminal window:

```bash
cd rag-app/frontend
streamlit run app.py
```

The Streamlit UI will launch at `http://localhost:8501`.

---

## API Endpoints

### 1. Health Check
- **Endpoint:** `GET /health`
- **Response:** `{"status": "ok"}`

### 2. Upload Document
- **Endpoint:** `POST /documents/upload`
- **Payload:** Form data (`file`)

### 3. Query RAG
- **Endpoint:** `POST /query`
- **Payload:** `{"question": "..."}`
