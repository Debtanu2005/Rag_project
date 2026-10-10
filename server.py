import os
import sys
from typing import List

from dotenv import load_dotenv

# The Gemini agents are created at import time, so the API key from .env
# must be loaded before importing the pipeline
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import uvicorn
from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app
from pydantic import BaseModel

# Importing the pipeline loads the embedding model, reranker and agents once.
# They stay in memory for as long as this server runs.
from pipeline.pipe import ingest_documents, run_pipeline, get_reranker

get_reranker()

app = FastAPI(title="RAG Pipeline")

# Same port as the API, so Prometheus keeps scraping localhost:8000/metrics
app.mount("/metrics", make_asgi_app())


class IngestRequest(BaseModel):
    pdf_paths: List[str]


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def status():
    return {"status": "running", "endpoints": ["POST /ingest", "POST /query", "GET /metrics"]}


@app.post("/ingest")
def ingest(request: IngestRequest):
    try:
        new_files = ingest_documents(request.pdf_paths)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    status = "ingested" if new_files else "already ingested"
    return {"status": status, "pdf_paths": request.pdf_paths}


@app.post("/query")
def query(request: QueryRequest):
    try:
        answer = run_pipeline(request.query)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    print(f"\nQuestion: {request.query}\nAnswer: {answer}\n")
    return {"query": request.query, "answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
