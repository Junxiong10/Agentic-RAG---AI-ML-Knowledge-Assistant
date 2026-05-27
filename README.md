# 🤖 Agentic RAG
**100% Local: Qwen 2.5:7b + nomic-embed-text via Ollama. No OpenAI, no TensorFlow.**

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed

---

## Setup (4 steps)

### 1. Pull both Ollama models
```bash
ollama pull qwen2.5:7b          # LLM for answering
ollama pull nomic-embed-text    # Embedding model (small & fast)
ollama serve                    # Keep running in a separate terminal
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python -m streamlit run app.py
```

Open http://localhost:8501

---

## Run tests
```bash
python test_cases.py
```

---

## Project Structure

```
agentic_rag_simple/
├── app.py             # Streamlit UI
├── agent.py           # LangGraph: retrieve → grade → generate → grade
├── ingestion.py       # Load → chunk → embed (Ollama) → ChromaDB + BM25
├── retrieval.py       # Hybrid search: Dense + BM25 + RRF fusion
├── test_cases.py      # 10 test cases
├── requirements.txt
└── knowledge_base/    # 10 AI & ML .txt documents
```

---

## Agent Flow

```
Question
   ↓
Retrieve (Dense ChromaDB + BM25 → RRF fusion)
   ↓
Grade Documents (Qwen checks each chunk for relevance)
   ↓ not enough → Rewrite Query → Retrieve again (max 2 retries)
   ↓ enough
Generate Answer + Citations [KB-00X]
   ↓
Grade Answer (Qwen checks if question resolved)
   ↓ good → Return Answer + Citations + Trace
```

---

## Models Used

| Model | Purpose |
|---|---|
| `qwen2.5:7b` | LLM — answering, grading, rewriting |
| `nomic-embed-text` | Embeddings — fast, accurate, 768-dim |

Both run 100% locally via Ollama. No API keys needed.

---

## Knowledge Base

| ID | Topic |
|---|---|
| KB-001 | Machine Learning Fundamentals |
| KB-002 | Deep Learning & Neural Networks |
| KB-003 | Natural Language Processing |
| KB-004 | Retrieval Augmented Generation |
| KB-005 | Agentic AI Systems |
| KB-006 | Computer Vision |
| KB-007 | Reinforcement Learning |
| KB-008 | AI Ethics & Responsible AI |
| KB-009 | Generative AI & LLMs |
| KB-010 | MLOps & AI in Production |
