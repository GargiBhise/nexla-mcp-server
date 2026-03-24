import os
import sys
import json
import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from src.metadata import extract_metadata

# Local embedding model (no API key needed)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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
    """Generate embeddings for all chunks using sentence-transformers (local, no API key)."""
    # Extract just the text from each chunk
    texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings locally using all-MiniLM-L6-v2
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    return np.array(embeddings, dtype=np.float32)


def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS index from the embeddings array."""
    # Get the dimension of the embeddings (384 for all-MiniLM-L6-v2)
    dimension = embeddings.shape[1]

    # Create a flat L2 (Euclidean distance) index
    index = faiss.IndexFlatL2(dimension)

    # Add all embeddings to the index
    index.add(embeddings)

    return index


def ingest_documents(data_dir: str) -> tuple:
    """
    Main entry point. Finds all PDFs, parses, chunks, embeds, and builds FAISS index.
    Returns (faiss_index, chunks_list, metadata_dict).
    """
    # Step 1: Find all PDF files
    pdf_paths = _find_pdfs(data_dir)
    # Use stderr for logging — stdout is reserved for MCP JSON-RPC messages
    print(f"Found {len(pdf_paths)} PDFs", file=sys.stderr)

    all_chunks = []
    all_metadata = {}

    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        print(f"Processing: {filename}", file=sys.stderr)

        # Step 2: Extract metadata for this document
        all_metadata[filename] = extract_metadata(pdf_path, filename)

        # Step 3: Parse pages from the PDF
        pages = _parse_pdf(pdf_path)

        # Step 4: Chunk each page's text
        for page_data in pages:
            chunks = _chunk_text(page_data["text"], filename, page_data["page"])
            all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}", file=sys.stderr)

    # Step 5: Generate embeddings for all chunks
    embeddings = _embed_chunks(all_chunks)

    # Step 6: Build the FAISS index
    index = _build_faiss_index(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors", file=sys.stderr)

    return index, all_chunks, all_metadata
