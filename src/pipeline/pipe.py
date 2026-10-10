import os
import sys
import time
from functools import lru_cache
from typing import List, TypedDict

# Project modules import each other as `utils` (project root) and
# `data_stores`, `agents`, ... (src/), so both folders must be on the path.
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC_DIR)
for path in (ROOT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langgraph.graph import StateGraph, START, END

from document_loders.pdf_loders import PDFLoader, vector_store, bm25_store
from agents.formal_agent import grade_query
from agents.rag_agent import answer_query
from reranker import HybridReranker
from monitoring.metrics import (
    REQUESTS,
    REQUEST_LATENCY,
    TTFT,
    TOKENS_PER_SECOND,
    OUTPUT_TOKENS,
    NODE_LATENCY,
)


@lru_cache(maxsize=1)
def get_reranker():
    # Cross-encoder weights are loaded on first use and reused for every query
    return HybridReranker(device="cuda" if torch.cuda.is_available() else "cpu")


# PDFs already ingested in this process and their chunks. The BM25 index lives
# in memory, so it is rebuilt from these chunks on every new run.
_ingested_files = set()
_ingested_chunks: List[Document] = []


def _load_chunks(pdf_path: str) -> List[Document]:
    # Chroma persists to disk: if this PDF was embedded in an earlier run,
    # reuse its chunks instead of running OCR and embedding again
    stored = vector_store.client.get(where={"source": pdf_path})
    if stored["documents"]:
        print(f"Using {len(stored['documents'])} stored chunks for {pdf_path}")
        return [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(stored["documents"], stored["metadatas"])
        ]

    print(f"Uploading {pdf_path} to vector store...")
    chunks = PDFLoader(pdf_path).load_and_process_pdf()
    vector_store.client.add_documents(chunks)
    return chunks


def ingest_documents(pdf_paths: List[str]) -> List[str]:
    # Each PDF is ingested once per server run; repeat calls are a no-op.
    # Returns the PDFs that were newly ingested by this call.
    new_files = [path for path in pdf_paths if path not in _ingested_files]
    if not new_files:
        return []

    for pdf_path in new_files:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        _ingested_chunks.extend(_load_chunks(pdf_path))
        _ingested_files.add(pdf_path)

    # BM25Retriever cannot add documents incrementally, so rebuild it over all chunks
    bm25_store.client = BM25Retriever.from_documents(_ingested_chunks, k=bm25_store.k)
    return new_files


class RAGState(TypedDict, total=False):
    query: str
    is_relevant: bool
    docs: List[Document]
    ranked_docs: List[Document]
    answer: str


def grade_user_query(state: RAGState):
    relevant = grade_query(state["query"])
    return {"is_relevant": relevant.strip().lower().startswith("yes")}


def check_relevance(state: RAGState):
    return "retrieve" if state["is_relevant"] else "reject_query"


def reject_query(state: RAGState):
    return {"answer": "Please ask a question related to the uploaded documents."}


def retrieve(state: RAGState):
    query = state["query"]

    semantic_docs = vector_store.client.similarity_search(query, k=5)
    keyword_docs = bm25_store.query(query, n_results=5)

    # Merge both result lists, dropping chunks returned by both retrievers
    seen = set()
    docs = []
    for doc in semantic_docs + keyword_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            docs.append(doc)

    return {"docs": docs}


def rerank(state: RAGState):
    return {"ranked_docs": get_reranker().rerank(state["query"], state["docs"])}


def generate_answer(state: RAGState):
    return {"answer": answer_query(state["query"], state["ranked_docs"])}


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("grade_query", grade_user_query)
    graph.add_node("reject_query", reject_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate_answer", generate_answer)

    graph.add_edge(START, "grade_query")
    graph.add_conditional_edges("grade_query", check_relevance, ["retrieve", "reject_query"])
    graph.add_edge("reject_query", END)
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


rag_graph = build_graph()


def run_pipeline(query: str) -> str:
    # Documents must be uploaded with ingest_documents() before querying
    if not _ingested_files:
        raise RuntimeError("No documents uploaded. Call ingest_documents() first.")

    start = time.perf_counter()
    last_node_end = start
    first_token_at = None
    chunk_count = 0
    output_tokens = 0
    final_state = {}

    try:
        # "messages" streams LLM tokens from inside nodes (used for TTFT and
        # tokens/sec), "updates" fires as each node finishes (node latency)
        for mode, payload in rag_graph.stream(
            {"query": query}, stream_mode=["messages", "updates"]
        ):
            now = time.perf_counter()

            if mode == "updates":
                for node, update in payload.items():
                    NODE_LATENCY.labels(node=node).observe(now - last_node_end)
                    final_state.update(update or {})
                last_node_end = now
                continue

            chunk, metadata = payload
            if metadata.get("langgraph_node") != "generate_answer":
                continue
            if not (chunk.content or getattr(chunk, "tool_call_chunks", None)):
                continue

            if first_token_at is None:
                first_token_at = now
            chunk_count += 1
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                output_tokens += usage.get("output_tokens", 0)
    except Exception:
        REQUESTS.labels(status="error").inc()
        REQUEST_LATENCY.observe(time.perf_counter() - start)
        raise

    end = time.perf_counter()
    REQUEST_LATENCY.observe(end - start)

    if not final_state.get("is_relevant"):
        REQUESTS.labels(status="rejected").inc()
    else:
        REQUESTS.labels(status="answered").inc()

    if first_token_at is not None:
        TTFT.observe(first_token_at - start)

        # Fall back to counting stream chunks if the model reports no usage
        tokens = output_tokens or chunk_count
        OUTPUT_TOKENS.inc(tokens)
        generation_time = end - first_token_at
        if generation_time > 0:
            TOKENS_PER_SECOND.observe(tokens / generation_time)

    return final_state["answer"]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/pipeline/pipe.py <query> <pdf_path> [<pdf_path> ...]")
        sys.exit(1)

    ingest_documents(sys.argv[2:])
    print(run_pipeline(sys.argv[1]))
