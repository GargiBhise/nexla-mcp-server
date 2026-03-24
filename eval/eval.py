import os
import json
from src.ingest import ingest_documents
from src.retriever import retrieve
from src.answerer import generate_answer

# Data directory containing the PDFs and JSONL files
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _load_qa_pairs(data_dir: str) -> list[dict]:
    """Load all Q&A pairs from JSONL files across all data subdirectories."""
    qa_pairs = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jsonl"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            qa_pairs.append(json.loads(line))
    return qa_pairs
