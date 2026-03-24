import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Model to use for answer generation
MODEL = "claude-sonnet-4-20250514"


def generate_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generate a grounded answer using Claude based on retrieved chunks.
    - Takes the question and retrieved chunks from the retriever
    - Builds a context string with source labels (filename + page)
    - Sends it to Claude with a system prompt that says 'only use provided context'
    - Returns a dict with the answer text and a deduplicated list of sources
    """
    # Build context from retrieved chunks with source labels
    context = ""
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context += f"\n--- Source {i}: {chunk['filename']}, Page {chunk['page']} ---\n"
        context += chunk["text"] + "\n"

    # System prompt constrains Claude to only use provided context
    system_prompt = (
        "You are a document Q&A assistant. Answer the user's question using ONLY "
        "the provided context below. Do not use any outside knowledge. "
        "If the answer cannot be found in the context, say so clearly. "
        "Always cite which source document and page your answer came from."
    )

    # Send question + context to Claude
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        ],
    )

    # Extract the answer text
    answer_text = message.content[0].text

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
