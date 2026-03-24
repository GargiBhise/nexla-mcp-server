import os
import json
import nltk
from nltk.corpus import stopwords
from src.ingest import ingest_documents
from src.retriever import retrieve
from src.answerer import generate_answer

# Download stop words corpus if not already present
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


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


def run_evaluation():
    """Run all Q&A pairs against the system and report accuracy by question type."""
    # Step 1: Ingest documents
    print("Ingesting documents...")
    index, chunks, metadata = ingest_documents(DATA_DIR)

    # Step 2: Load ground-truth Q&A pairs
    qa_pairs = _load_qa_pairs(DATA_DIR)
    print(f"Loaded {len(qa_pairs)} Q&A pairs")

    # Track results by question type
    results = {}

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        expected = qa["answer"]
        q_type = qa.get("type", "unknown")

        # Initialize tracking for this question type
        if q_type not in results:
            results[q_type] = {"total": 0, "correct": 0}
        results[q_type]["total"] += 1

        # Get the system's answer
        retrieved = retrieve(question, index, chunks)
        response = generate_answer(question, retrieved)
        actual = response["answer"]

        # Detect refusal — LLM is instructed to say exactly "I don't know the answer"
        actual_lower = actual.lower()
        is_refusal = "i don't know the answer" in actual_lower

        # Check if key content words from expected answer appear in actual answer
        # Filter out NLTK stop words so common terms don't inflate the match ratio
        expected_words = set(expected.lower().split()) - STOP_WORDS
        actual_words = set(actual.lower().split()) - STOP_WORDS
        # Count how many expected content words appear in the actual answer
        matches = sum(1 for word in expected_words if word in actual_words)
        match_ratio = matches / len(expected_words) if expected_words else 0
        is_correct = match_ratio >= 0.5 and not is_refusal
        if is_correct:
            results[q_type]["correct"] += 1

        # Print progress with details
        status = "PASS" if is_correct else "FAIL"
        print(f"\n[{i+1}/{len(qa_pairs)}] [{status}] ({q_type}) {question[:80]}")
        print(f"  Expected: {expected[:100]}")
        print(f"  Actual:   {actual[:100]}")
        refusal_tag = " [REFUSAL DETECTED]" if is_refusal else ""
        print(f"  Match:    {matches}/{len(expected_words)} content words ({match_ratio:.0%}){refusal_tag}")

    # Step 3: Print summary
    print("\n--- Evaluation Summary ---")
    for q_type, counts in results.items():
        accuracy = (counts["correct"] / counts["total"]) * 100 if counts["total"] > 0 else 0
        print(f"{q_type}: {counts['correct']}/{counts['total']} ({accuracy:.1f}%)")


if __name__ == "__main__":
    run_evaluation()
