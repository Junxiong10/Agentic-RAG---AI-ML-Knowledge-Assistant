"""
retrieval.py - Hybrid retrieval: Dense + BM25 fused with RRF
"""

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever


def dense_search(query, vectorstore, k):
    return vectorstore.similarity_search(query, k=k)


def sparse_search(query, bm25, k):
    bm25.k = k
    return bm25.invoke(query)


def rrf_fusion(lists, k=60):
    scores, doc_map = {}, {}
    for ranked in lists:
        for rank, doc in enumerate(ranked):
            key = doc.metadata.get("chunk_id", doc.page_content[:60])
            scores[key]  = scores.get(key, 0.0) + 1.0 / (rank + 1 + k)
            doc_map[key] = doc
    return [doc_map[k] for k in sorted(scores, key=lambda x: scores[x], reverse=True)]


from sentence_transformers import CrossEncoder

_CROSS_ENCODER = None

def get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        _CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    return _CROSS_ENCODER

def hybrid_retrieve(query, vectorstore, bm25, top_k=3):
    import concurrent.futures
    # Retrieve more candidates for the re-ranker
    candidate_k = max(15, top_k * 3)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_dense = executor.submit(dense_search, query, vectorstore, k=candidate_k)
        future_sparse = executor.submit(sparse_search, query, bm25, k=candidate_k)
        dense = future_dense.result()
        sparse = future_sparse.result()
        
    # Fuse them using our weighted RRF
    fused_docs = rrf_fusion([dense, sparse])[:candidate_k]
    
    if not fused_docs:
        return []
        
    # Cross-Encoder Re-ranking
    encoder = get_cross_encoder()
    pairs = [[query, doc.page_content] for doc in fused_docs]
    scores = encoder.predict(pairs)
    
    # Sort by the new score
    scored_docs = list(zip(scores, fused_docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    return [doc for score, doc in scored_docs[:top_k]]


def format_context(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        parts.append(
            f"[{i}] {meta.get('title','?')} | {meta.get('chunk_id','?')}\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)
