import os
import json
import faiss
import numpy as np
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
from src.metadata import extract_metadata

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embedding model to use
EMBEDDING_MODEL = "text-embedding-3-small"

# Chunk size and overlap in characters
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def _find_pdfs(data_dir: str) -> list[str]:
    """Recursively find all PDF files in the data directory."""
    pdf_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file))
    return pdf_paths


def _parse_pdf(pdf_path: str) -> list[dict]:
    """Extract text and tables from each page of a PDF."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract plain text from the page
            text = page.extract_text() or ""

            # Extract tables and append as tab-separated text
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    row_text = "\t".join(cell for cell in row if cell)
                    text += "\n" + row_text

            # Only include pages that have content
            if text.strip():
                pages.append({"page": page_num, "text": text.strip()})

    return pages
