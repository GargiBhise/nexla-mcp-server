import requests

# Ollama API endpoint (local)
OLLAMA_URL = "http://localhost:11434/api/generate"

# Model to use for answer generation
MODEL = "llama3.2"


def generate_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generate a grounded answer using Ollama (local LLM) based on retrieved chunks.
    - Takes the question and retrieved chunks from the retriever
    - Builds a context string with source labels (filename + page)
    - Sends it to Ollama with a prompt that says 'only use provided context'
    - Returns a dict with the answer text and a deduplicated list of sources
    """
    # Build context from retrieved chunks with source labels
    context = ""
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context += f"\n--- Source {i}: {chunk['filename']}, Page {chunk['page']} ---\n"
        context += chunk["text"] + "\n"

    # Prompt constrains the LLM to only use provided context
    prompt = (
        "You are a document Q&A assistant. Answer the user's question using ONLY "
        "the provided context below. Do not use any outside knowledge. "
        "If the answer cannot be found in the context, say so clearly. "
        "Always cite which source document and page your answer came from.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    # Send prompt to Ollama local API
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    })

    # Extract the answer text
    answer_text = response.json()["response"]

    # Build source attribution list
    sources = []
    seen = set()
    for chunk in retrieved_chunks:
        key = (chunk["filename"], chunk["page"])
        if key not in seen:
            sources.append({
                "filename": chunk["filename"],
                "page": chunk["page"],
                "excerpt": chunk["text"][:200],
            })
            seen.add(key)

    return {
        "answer": answer_text,
        "sources": sources,
    }
