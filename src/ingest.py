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


def _chunk_text(text: str, filename: str, page: int) -> list[dict]:
    """Split text into overlapping chunks with source metadata."""
    chunks = []
    start = 0
    while start < len(text):
        # Slice the text from start to start + chunk size
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]

        # Attach source metadata to each chunk
        chunks.append({
            "text": chunk_text,
            "filename": filename,
            "page": page,
        })

        # Move forward by chunk size minus overlap
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def _embed_chunks(chunks: list[dict]) -> np.ndarray:
    """Generate embeddings for all chunks using OpenAI API."""
    # Extract just the text from each chunk
    texts = [chunk["text"] for chunk in chunks]

    # Call OpenAI embeddings API in a single batch request
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    # Convert response to numpy array for FAISS
    embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)

    return embeddings


def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS index from the embeddings array."""
    # Get the dimension of the embeddings (1536 for text-embedding-3-small)
    dimension = embeddings.shape[1]

    # Create a flat L2 (Euclidean distance) index
    index = faiss.IndexFlatL2(dimension)

    # Add all embeddings to the index
    index.add(embeddings)

    return index
