import os
import sys
import time

from dotenv import load_dotenv

# The Gemini agents are created at import time, so the API key from .env
# must be loaded before importing the pipeline
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from pipeline.pipe import ingest_documents, run_pipeline
from monitoring.metrics import start_metrics_server

QUERY = "What is naive bayes algorithm?"
PDF_PATH = r"C:\Users\aspir\Desktop\rag_practice\Rag_project\ML_lecture_5_annotated.pdf"

# Serve TTFT, latency and tokens/sec metrics for Prometheus
start_metrics_server(port=8000)

# 1. Ingest the document (OCR + embedding only on the first run; later runs
#    reuse the chunks already stored in chroma_db)
ingest_documents([PDF_PATH])

# 2. Answer the query
answer = run_pipeline(QUERY)
print(f"\nQuestion: {QUERY}")
print(f"Answer: {answer}")

# 3. Keep the metrics endpoint alive so Prometheus can scrape this run
time.sleep(10)
