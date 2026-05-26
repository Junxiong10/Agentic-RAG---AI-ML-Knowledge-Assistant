"""
app.py — Streamlit UI for Agentic RAG (Qwen 2.5:7b via Ollama)
Run: streamlit run app.py
"""
# python -m streamlit run app.py

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import re
import streamlit as st
from ingestion import build_indexes
from agent import build_graph, ask

st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="wide")

st.title("🤖 Agentic RAG — AI & ML Knowledge Assistant")
st.caption("Powered by Qwen 2.5:7b (Ollama) + ChromaDB + BM25 + LangGraph")


# ── Citation badge renderer ────────────────────────────────

def render_answer(answer: str) -> str:
    """
    Replace [KB-001] tags in the answer with styled HTML badge spans.
    e.g. [KB-001] → <span class="cite-badge">KB-001</span>
    """
    def badge(m):
        kid = m.group(1)
        return (
            f'<span style="display:inline-block;background:#1e3a5f;color:#90cdf4;'
            f'font-size:11px;font-weight:700;padding:1px 6px;border-radius:10px;'
            f'margin:0 2px;vertical-align:middle;">{kid}</span>'
        )
    return re.sub(r'\[(KB-\d{3})\]', badge, answer)


# ── Initialize ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base & building indexes...")
def init():
    vectorstore, bm25 = build_indexes()
    graph = build_graph(vectorstore, bm25)
    return graph

graph = init()

# ── Sidebar ────────────────────────────────────────────────

with st.sidebar:
    st.header("📚 Knowledge Base")
    st.markdown("""
- KB-001 · ML Fundamentals
- KB-002 · Deep Learning
- KB-003 · NLP
- KB-004 · RAG Systems
- KB-005 · AI Agents
- KB-006 · Computer Vision
- KB-007 · Reinforcement Learning
- KB-008 · AI Ethics
- KB-009 · Generative AI & LLMs
- KB-010 · MLOps
""")
    st.divider()
    st.header("⚙️ Settings")
    show_trace  = st.toggle("Show agent trace",      value=True)
    show_chunks = st.toggle("Show retrieved chunks", value=False)

    st.divider()
    st.header("💡 Example Questions")
    examples = [
        "What are the applications of machine learning?",
        "What is RAG and why is it needed?",
        "How does Agentic RAG improve upon traditional RAG?",
        "What is RLHF?",
        "Explain the Transformer architecture.",
        "What is the EU AI Act?",
        "What is overfitting?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["input"] = ex

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Chat history ────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # Render with citation badges
            st.markdown(render_answer(msg["content"]), unsafe_allow_html=True)
            if "state" in msg:
                s = msg["state"]
                if show_trace and s.get("trace"):
                    with st.expander("🔄 Agent trace", expanded=False):
                        for step in s["trace"]:
                            st.markdown(f"`{step}`")
                if s.get("citations"):
                    st.markdown("**📎 Sources cited:**")
                    for c in s["citations"]:
                        st.markdown(f"- `{c['doc_id']}` — {c['title']}")
                if show_chunks and s.get("docs"):
                    with st.expander(f"🔍 Retrieved chunks ({len(s['docs'])})", expanded=False):
                        for i, doc in enumerate(s["docs"], 1):
                            st.markdown(f"**Chunk {i}** · `{doc.metadata.get('chunk_id','?')}` · {doc.metadata.get('title','?')}")
                            st.caption(doc.page_content[:300] + "...")
                            st.divider()
        else:
            st.markdown(msg["content"])

# ── Input ───────────────────────────────────────────────────

question = st.chat_input("Ask anything about AI & Machine Learning...")

if "input" in st.session_state:
    question = st.session_state.pop("input")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Agent thinking..."):
            state = ask(question, graph)

        answer = state.get("answer", "No answer generated.")

        # Render answer with inline citation badges
        st.markdown(render_answer(answer), unsafe_allow_html=True)

        if show_trace and state.get("trace"):
            with st.expander("🔄 Agent trace", expanded=True):
                for step in state["trace"]:
                    st.markdown(f"`{step}`")

        if state.get("citations"):
            st.markdown("**📎 Sources cited:**")
            for c in state["citations"]:
                st.markdown(f"- `{c['doc_id']}` — {c['title']}")

        if show_chunks and state.get("docs"):
            with st.expander(f"🔍 Retrieved chunks ({len(state['docs'])})", expanded=False):
                for i, doc in enumerate(state["docs"], 1):
                    st.markdown(f"**Chunk {i}** · `{doc.metadata.get('chunk_id','?')}` · {doc.metadata.get('title','?')}")
                    st.caption(doc.page_content[:300] + "...")
                    st.divider()

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "state":   state,
    })
