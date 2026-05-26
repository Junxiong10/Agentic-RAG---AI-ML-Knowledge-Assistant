"""
agent.py
Agentic RAG — strict grounding, no hallucination, verified citations.
Always retrieves 3 chunks. Token limit scales with answer complexity.
"""

import re
import operator
from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from retrieval import hybrid_retrieve

OLLAMA_MODEL = "qwen2.5:7b"


_LLM_INSTANCE = None

def get_llm():
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        _LLM_INSTANCE = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0,
            num_predict=1000,  # enough for any complete answer
            num_ctx=4096,
        )
    return _LLM_INSTANCE


# ── State ──────────────────────────────────────────────────

def _replace(_, new): return new

class State(TypedDict):
    question:  str
    queries:   Annotated[List[str], _replace]
    docs:      Annotated[List[Document], _replace]
    answer:    Annotated[str, _replace]
    citations: Annotated[List[dict], _replace]
    retries:   Annotated[int, _replace]
    retry:     Annotated[bool, _replace]
    trace:     Annotated[List[str], operator.add]


# ── Prompt ─────────────────────────────────────────────────

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict knowledge base assistant.

RULES:
1. Use ONLY information from the SOURCES below. Nothing from your training data.
2. End every sentence with its source ID in brackets: [KB-001]
3. Include ALL relevant points from the sources — do not stop early or summarise.
4. If not in sources: say "This information is not available in my knowledge base."
5. Do not infer, guess, or add anything beyond the source text.

SOURCES:
{context}"""),
    ("human", "{question}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite as a specific search query. Output ONLY the query."),
    ("human", "{question}"),
])

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a search query planner.
If the user's question is complex and asks to compare or explain multiple distinct concepts, break it down into 2 to 3 specific search queries.
If the question is simple, just output the question as a single query.
Output ONLY the queries, one per line. Do NOT include bullet points, numbers, or any other text."""),
    ("human", "{question}"),
])


# ── Context — full chunk text, no trimming ─────────────────

def build_context(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        meta   = doc.metadata
        doc_id = meta.get("doc_id", "?")
        title  = meta.get("title", "?")
        parts.append(f"[{doc_id}] {title}:\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


# ── Citation helpers ───────────────────────────────────────

def verify_citations(answer: str, docs: List[Document]) -> str:
    """Remove any citation tag not in the retrieved doc set."""
    retrieved_ids = {doc.metadata.get("doc_id", "") for doc in docs}
    def check(m):
        kid = m.group(1)
        return f"[{kid}]" if kid in retrieved_ids else ""
    return re.sub(r'\[(KB-\d{3})\]', check, answer)


def extract_citations(answer: str, docs: List[Document]) -> List[dict]:
    """Extract cited sources. Fallback to word-overlap if model forgot tags."""
    retrieved = {doc.metadata.get("doc_id", ""): doc for doc in docs}
    citations, seen = [], set()

    # Pass 1: explicit tags
    for did in set(re.findall(r'KB-\d{3}', answer)):
        if did in retrieved and did not in seen:
            citations.append({"doc_id": did, "title": retrieved[did].metadata.get("title", did)})
            seen.add(did)

    # Pass 2: word overlap fallback
    if not citations:
        stopwords = {"which", "these", "their", "about", "where", "there",
                     "other", "within", "between", "through", "would", "could",
                     "should", "using", "based", "learn", "model", "train"}
        answer_lower = answer.lower()
        for did, doc in retrieved.items():
            if did in seen:
                continue
            words = set(w for w in re.findall(r'\b[a-z]{5,}\b', doc.page_content.lower())
                        if w not in stopwords)
            if sum(1 for w in words if w in answer_lower) >= 8:
                citations.append({"doc_id": did, "title": doc.metadata.get("title", did)})
                seen.add(did)

    return citations


def clean_answer(answer: str) -> str:
    answer = re.sub(r'\[\[?(KB-\d{3})\]?\]', r'[\1]', answer)
    answer = re.sub(r'([.!?:])(\[KB-\d{3}\])', r'\1 \2', answer)
    answer = re.sub(r'  +', ' ', answer)
    return answer.strip()


# ── Nodes ──────────────────────────────────────────────────

def node_decompose(state: State) -> dict:
    chain = DECOMPOSE_PROMPT | get_llm() | StrOutputParser()
    raw = chain.invoke({"question": state["question"]}).strip()
    queries = [q.strip() for q in raw.split("\n") if q.strip()]
    if not queries:
        queries = [state["question"]]
    return {
        "queries": queries,
        "trace": [f"🧩 Planned {len(queries)} queries: {queries}"]
    }


def node_retrieve(state: State, vectorstore, bm25) -> dict:
    queries = state.get("queries")
    if not queries:
        queries = [state["question"]]
        
    all_docs = []
    seen = set()
    k_per_query = 3 if len(queries) == 1 else 2
    
    import concurrent.futures
    
    def fetch(q):
        return hybrid_retrieve(q, vectorstore, bm25, top_k=k_per_query)
        
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch, queries))
        
    for docs in results:
        for d in docs:
            did = d.metadata.get("chunk_id", d.metadata.get("doc_id", ""))
            if did not in seen:
                seen.add(did)
                all_docs.append(d)
                
    return {
        "docs":  all_docs,
        "trace": [f"📚 Retrieved {len(all_docs)} unique chunks across {len(queries)} queries"],
    }


def node_generate(state: State) -> dict:
    docs    = state["docs"]
    context = build_context(docs)
    chain   = ANSWER_PROMPT | get_llm() | StrOutputParser()
    raw     = chain.invoke({"question": state["question"], "context": context})

    verified  = verify_citations(raw, docs)
    answer    = clean_answer(verified)
    citations = extract_citations(answer, docs)

    no_info      = "not available in my knowledge base" in answer.lower()
    should_retry = no_info and state["retries"] < 1

    if no_info and not should_retry:
        answer = "This information is not available in my knowledge base."
        citations = []
        docs = []
        trace_msgs = ["💬 Generated (no information found)"]
    else:
        trace_msgs = [
            f"💬 Generated ({len(answer.split())} words)",
            f"📎 Verified sources: {[c['doc_id'] for c in citations]}"
            + (" — retrying..." if should_retry else ""),
        ]

    result = {
        "answer":    answer,
        "citations": citations,
        "retry":     should_retry,
        "trace":     trace_msgs,
    }
    
    if no_info and not should_retry:
        result["docs"] = docs

    return result


def node_rewrite(state: State) -> dict:
    chain = REWRITE_PROMPT | get_llm() | StrOutputParser()
    new_q = chain.invoke({"question": state["question"]}).strip()
    return {
        "queries": [new_q],
        "retries": state["retries"] + 1,
        "retry":   False,
        "trace":   [f"✏️  Rewritten query: \"{new_q}\""],
    }


# ── Routing ────────────────────────────────────────────────

def route(state: State):
    return "rewrite" if state.get("retry") else END


# ── Graph ──────────────────────────────────────────────────

def build_graph(vectorstore, bm25):
    def _decompose(s): return node_decompose(s)
    def _retrieve(s): return node_retrieve(s, vectorstore, bm25)
    def _generate(s): return node_generate(s)
    def _rewrite(s):  return node_rewrite(s)

    g = StateGraph(State)
    g.add_node("decompose", _decompose)
    g.add_node("retrieve", _retrieve)
    g.add_node("generate", _generate)
    g.add_node("rewrite",  _rewrite)

    g.set_entry_point("decompose")
    g.add_edge("decompose", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_conditional_edges("generate", route, {END: END, "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")

    return g.compile()


def ask(question: str, graph) -> dict:
    return graph.invoke({
        "question":  question,
        "queries":   [],
        "docs":      [],
        "answer":    "",
        "citations": [],
        "retries":   0,
        "retry":     False,
        "trace":     [],
    })
