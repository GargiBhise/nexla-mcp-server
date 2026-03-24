import numpy as np
from sentence_transformers import SentenceTransformer

# Same model used during ingestion
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(question: str, index, chunks: list[dict], k: int = 5) -> list[dict]:
    """Search FAISS index for the top-k chunks most relevant to the question."""
    # Embed the question using the same local model used during ingestion
    query_embedding = embedding_model.encode([question])
    query_embedding = np.array(query_embedding, dtype=np.float32)

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
