import os
import sys
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
    # Use stderr for logging — stdout is reserved for MCP JSON-RPC messages
    print("Starting document ingestion...", file=sys.stderr)
    try:
        index, chunks, metadata = ingest_documents(DATA_DIR)
        print(f"Server ready. {len(chunks)} chunks indexed from {len(metadata)} documents.", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Data directory not found at {DATA_DIR}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Error during ingestion: {e}", file=sys.stderr)
        raise


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


@mcp.tool()
def get_document_metadata(filename: str) -> dict:
    """Return metadata for a specific document — title, authors, page count, word count, reference count."""
    if filename not in metadata:
        return {"error": f"Document '{filename}' not found. Use list_documents() to see available files."}
    return metadata[filename]


if __name__ == "__main__":
    # Run ingestion before starting the server
    startup()
    # Start the MCP server using stdio transport
    mcp.run(transport="stdio")
