import os
import json
from src.ingest import ingest_documents
from src.retriever import retrieve
from src.answerer import generate_answer

# Data directory containing the PDFs and JSONL files
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
