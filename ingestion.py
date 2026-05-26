"""
ingestion.py - Section-aware chunking using correct separator.
"""

import pickle
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings

KB_DIR        = "./knowledge_base"
CHROMA_DIR    = "./chroma_db"
BM25_PATH     = "./bm25.pkl"
CHUNK_SIZE    = 3000
CHUNK_OVERLAP = 200
EMBED_MODEL   = "nomic-embed-text"


def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)


def split_text(text: str):
    """
    Split on the actual separator used in KB files: double-newline around ---.
    Keeps each named section (OVERVIEW, TYPES OF..., etc.) as one chunk.
    Falls back to character split only if a section exceeds CHUNK_SIZE.
    """
    chunks = []
    sections = text.split("\n\n---\n\n")   # exact separator from KB files
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= CHUNK_SIZE:
            chunks.append(section)
        else:
            # Section too big — split by character with overlap
            start = 0
            while start < len(section):
                chunks.append(section[start:start + CHUNK_SIZE])
                start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_documents():
    docs = []
    for fpath in sorted(Path(KB_DIR).glob("*.txt")):
        if fpath.name.startswith("00_"):
            continue
        text = fpath.read_text(encoding="utf-8")
        meta = {"source": fpath.name, "doc_id": "UNKNOWN", "title": fpath.name}
        for line in text.splitlines()[:8]:
            if line.startswith("TITLE:"):
                meta["title"] = line.replace("TITLE:", "").strip()
            elif line.startswith("DOC_ID:"):
                meta["doc_id"] = line.replace("DOC_ID:", "").strip()
        docs.append(Document(page_content=text, metadata=meta))
        print(f"  Loaded: {meta['doc_id']} — {meta['title']}")
    return docs


def chunk_documents(docs):
    chunks = []
    for doc in docs:
        for i, part in enumerate(split_text(doc.page_content)):
            chunks.append(Document(
                page_content=part,
                metadata={**doc.metadata, "chunk_id": f"{doc.metadata['doc_id']}-chunk{i+1}"},
            ))
    print(f"  Total chunks: {len(chunks)}")
    return chunks


def build_indexes(force=False):
    chroma_ready = Path(CHROMA_DIR).exists()
    bm25_ready   = Path(BM25_PATH).exists()

    if chroma_ready and bm25_ready and not force:
        print("[Ingestion] Loading existing indexes...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=get_embeddings(),
            collection_name="rag_kb",
        )
        with open(BM25_PATH, "rb") as f:
            bm25 = pickle.load(f)
        return vectorstore, bm25

    print("[Ingestion] Building indexes from scratch...")
    docs   = load_documents()
    chunks = chunk_documents(docs)

    print(f"  Embedding {len(chunks)} chunks via Ollama ({EMBED_MODEL})...")
    vectorstore = Chroma.from_documents(
        chunks, get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_name="rag_kb",
    )

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 6
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print("[Ingestion] Done.")
    return vectorstore, bm25
