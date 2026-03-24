import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"


def retrieve(question: str, index, chunks: list[dict], k: int = 5) -> list[dict]:
    """Search FAISS index for the top-k chunks most relevant to the question."""
    # Embed the question using the same model used during ingestion
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    )
    query_embedding = np.array([response.data[0].embedding], dtype=np.float32)

    # Search FAISS for the k nearest vectors
    distances, indices = index.search(query_embedding, k)

    # Collect matching chunks with their similarity scores
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            result = chunks[idx].copy()
            result["score"] = float(distances[0][i])
            results.append(result)

    return results
