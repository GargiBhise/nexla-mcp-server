import os
from fastmcp import FastMCP
from src.ingest import ingest_documents
from src.retriever import retrieve
from src.answerer import generate_answer

# Initialize the MCP server
mcp = FastMCP("nexla-doc-qa", description="Q&A over PDF documents with source attribution")

# Data directory containing the PDFs
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# These will be populated at startup
index = None
chunks = []
metadata = {}


def startup():
    """Run the ingestion pipeline and populate the global index, chunks, and metadata."""
    global index, chunks, metadata
    print("Starting document ingestion...")
    index, chunks, metadata = ingest_documents(DATA_DIR)
    print(f"Server ready. {len(chunks)} chunks indexed from {len(metadata)} documents.")


@mcp.tool()
def query_documents(question: str) -> dict:
    """
    Ask a natural language question across all indexed PDF documents.
    Returns a grounded answer with source attribution (filename and page).
    """
    # Find the most relevant chunks using FAISS
    retrieved = retrieve(question, index, chunks)

    # Generate a grounded answer using Claude
    result = generate_answer(question, retrieved)

    return result


@mcp.tool()
def list_documents() -> list[str]:
    """Return the list of all indexed PDF filenames."""
    return list(metadata.keys())
