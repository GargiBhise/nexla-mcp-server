# Nexla MCP Server — PDF Document Q&A

An MCP (Model Context Protocol) server that ingests PDF documents, indexes their content using vector embeddings, and exposes natural language Q&A tools to any MCP-compatible AI agent.

---

## 1. Setup Instructions

### Prerequisites

- Python 3.10+
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com/))

### Step 1: Clone and create virtual environment

```bash
git clone https://github.com/GargiBhise/FDE-AI-Take-Home-GargiBhise.git
cd FDE-AI-Take-Home-GargiBhise

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

This installs FastMCP, pdfplumber, sentence-transformers, FAISS, Anthropic SDK, NLTK, and python-dotenv. The sentence-transformers model (~90MB) downloads automatically on first run.

### Step 3: Configure environment

Create a `.env` file from the template:

```bash
cp .env.example .env            # macOS/Linux
copy .env.example .env          # Windows
```

Edit `.env` and add your Anthropic API key:

```
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

### Step 4: Run the evaluation (quickest way to see it work)

```bash
python -m eval.eval
```

This ingests all 5 PDFs, runs 30 ground-truth Q&A pairs against the system, and reports accuracy by question type. On first run, NLTK will automatically download the stopwords corpus (requires internet). You should see output like:

```
Ingesting documents...
Found 5 PDFs
Processing: P19-1598.pdf
...
Loaded 30 Q&A pairs

[1/30] [FAIL] (text-only) What is the primary challenge addressed by...
  Expected: The primary challenge addressed is incorporating factual...
  Actual:   The primary challenge addressed by the introduction of...
  Match:    8/19 content words (42%)
...
--- Evaluation Summary ---
text-only: 2/6 (33.3%)
multimodal-t: 6/14 (42.9%)
multimodal-f: 1/2 (50.0%)
meta-data: 0/8 (0.0%)
```

### Step 5: Start the MCP server

```bash
python -m src.server
```

The server ingests all PDFs at startup, then listens on stdio for MCP tool calls. You should see:

```
Starting document ingestion...
Server ready. 138 chunks indexed from 5 documents.
```

> **Note:** The server communicates via stdio (JSON-RPC), so it won't show a prompt. It's designed to be called by an MCP client like Claude Desktop, not used interactively in the terminal. Press `Ctrl+C` to stop.

### Step 6 (Optional): Connect to Claude Desktop

To use the server interactively via Claude Desktop, add the following to your config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "nexla-doc-qa": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/nexla-mcp-server"
    }
  }
}
```

---

## 2. Architecture Overview

### Pipeline

```
DATA (5 PDFs + 5 JSONL eval-only)
        |
        v
INGESTION (ingest.py + metadata.py)
  pdfplumber -> chunk (1500 chars, 200 overlap) -> sentence-transformers -> FAISS
        |
        v
STORAGE: FAISS index + chunks list + metadata dict (all in-memory)
        |
        v
RETRIEVAL (retriever.py)
  question -> embed -> FAISS top-5 -> chunks with metadata
        |
        v
ANSWER (answerer.py)
  chunks + question -> Claude API -> synthesized answer + sources
        |
        v
MCP SERVER (server.py — FastMCP + stdio)
  Tool 1: query_documents(question) -> {answer, sources}
  Tool 2: list_documents() -> [filenames]
  Tool 3: get_document_metadata(filename) -> {title, authors, counts}
```

### MCP Server

```
CLIENT (Claude Desktop / any MCP client)
    |
    |  stdio (JSON-RPC)
    |
    v
FASTMCP SERVER (server.py)
    |
    |--- query_documents(question)
    |       |-> retriever.py (embed + FAISS search)
    |       |-> answerer.py (Claude API)
    |       |-> return {answer, sources}
    |
    |--- list_documents()
    |       |-> metadata dict
    |       |-> return [filenames]
    |
    |--- get_document_metadata(filename)
            |-> metadata dict
            |-> return {title, authors, page_count, word_count, reference_count}
```

### How it works

1. **Ingestion** — At startup, `ingest.py` finds all PDFs in `data/`, extracts text and tables using pdfplumber, splits into overlapping chunks (1500 chars, 200 overlap), generates embeddings with sentence-transformers (`all-MiniLM-L6-v2`), and builds a FAISS vector index. `metadata.py` extracts document-level metadata (title, authors, page/word/reference counts).

2. **Retrieval** — When a question comes in, `retriever.py` embeds the question using the same model and performs a FAISS similarity search to find the top 5 most relevant chunks.

3. **Answer Generation** — `answerer.py` sends the retrieved chunks along with the question to Claude API. The system prompt instructs Claude to answer only from the provided context, give direct responses, and cite sources.

4. **MCP Server** — `server.py` uses FastMCP to expose 3 tools over stdio transport. Any MCP-compatible client (like Claude Desktop) can call these tools.

### Tech Stack

| Component | Choice | Reason |
|---|---|---|
| MCP Framework | FastMCP | Auto-generates tool schemas, simple stdio transport |
| PDF Parsing | pdfplumber | Text + table extraction from machine-readable PDFs |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Local, zero cost, no API key needed |
| Vector Store | FAISS (`faiss-cpu`) | Fast in-memory similarity search, prebuilt wheels |
| LLM | Claude API (`claude-sonnet-4-20250514`) | High-quality grounded answers with source citation |
| Transport | stdio | Local-only, Claude Desktop default |

### File Structure

```
src/
  server.py        # FastMCP server with 3 tools + startup ingestion
  ingest.py        # PDF parsing, chunking, embedding, FAISS index
  retriever.py     # FAISS similarity search
  answerer.py      # Claude API with RAG prompt
  metadata.py      # Title, authors, page/word/reference counts
eval/
  eval.py          # Accuracy evaluation by question type
examples/
  interactions.md  # Sample Q&A interactions
data/
  0-4/             # 5 PDF papers + 5 JSONL eval files
```

### Trade-offs

| Decision | Alternative | Why |
|---|---|---|
| FAISS over ChromaDB | ChromaDB | ChromaDB requires C++ build tools on Windows |
| Rebuild index at startup | Persist to disk | 5 small PDFs, takes seconds to re-embed |
| sentence-transformers over OpenAI | OpenAI embeddings | Local, zero cost, no API key for embeddings |
| Claude API over Ollama | Local Ollama | Ollama had timeout issues on local hardware |
| 50% word overlap eval | Exact string match | Strict matching was too rigid for natural language |

---

## 3. Tool Documentation

### Tool 1: `query_documents`

Ask a natural language question across all indexed PDF documents. Returns a grounded answer with source attribution (filename and page number).

**Input:**

| Parameter | Type | Description |
|---|---|---|
| `question` | `str` | Natural language question to answer from the documents |

**Output:**

```json
{
  "answer": "Participants classified texts into 3 categories: overt aggression, covert aggression, and non-aggression.",
  "sources": [
    {"filename": "W18-4401.pdf", "page": 1, "excerpt": "The data for the shared task..."}
  ]
}
```

**Example queries:**
- "How many categories of aggression were participants asked to classify texts into?"
- "What loss function achieved the highest average performance according to Table 5?"
- "What is the reported issue with sequence generation models in dialogue tasks?"

---

### Tool 2: `list_documents`

Returns the filenames of all indexed PDF documents. No parameters required.

**Input:** None

**Output:**

```json
["P19-1598.pdf", "W18-4401.pdf", "P19-1164.pdf", "D19-1539.pdf", "W18-5713.pdf"]
```

---

### Tool 3: `get_document_metadata`

Returns structured metadata for a specific document, including title, authors, and counts.

**Input:**

| Parameter | Type | Description |
|---|---|---|
| `filename` | `str` | Exact PDF filename (use `list_documents` to get valid names) |

**Output:**

```json
{
  "filename": "D19-1539.pdf",
  "title": "Cloze-driven Pretraining of Self-attention Networks",
  "authors": ["AlexeiBaevski,SergeyEdunov,YinhanLiu,LukeZettlemoyer,MichaelAuli"],
  "page_count": 10,
  "word_count": 4234,
  "reference_count": 35
}
```

> **Note:** Title and author extraction uses heuristics on PDF first-page text. Results are best-effort — multi-line titles may truncate and author names may not separate cleanly depending on PDF layout.

If the filename is not found, returns:

```json
{"error": "Document 'unknown.pdf' not found. Use list_documents() to see available files."}
```

---

## 4. Evaluation

The evaluation harness (`eval/eval.py`) tests the `query_documents` tool against 30 ground-truth Q&A pairs from JSONL files.

### Results

| Question Type | Score | Notes |
|---|---|---|
| text-only | 2/6 (33.3%) | General comprehension questions |
| multimodal-t | 6/14 (42.9%) | Table-based questions |
| multimodal-f | 1/2 (50.0%) | Figure-based questions (known limitation) |
| meta-data | 0/8 (0.0%) | Requires metadata tool, not RAG |

### Methodology

- **Stop word filtering:** NLTK English stop words are removed before comparison so common words don't inflate scores
- **Refusal detection:** The LLM is instructed to say "I don't know the answer" when context is insufficient; refusals are auto-marked as FAIL
- **Word overlap threshold:** At least 50% of content words from the expected answer must appear in the actual answer

### Known Limitations

- **Figure questions (multimodal-f):** Text-only RAG cannot extract information from figures or charts
- **Metadata questions:** The eval harness only tests `query_documents`, not `get_document_metadata`, so metadata questions score 0%
- **Index rebuilt at startup:** No persistence — a production system would cache embeddings to disk
- **Stateless:** No conversation memory between queries
- **Table parsing:** Best-effort via pdfplumber; complex table layouts may lose structure
- **Eval matching:** Word overlap can miss correct paraphrases or pass wrong-entity matches

---

## 5. Vibe Coding Reflection

### Tools Used

| Tool | Role |
|---|---|
| Claude Code (CLI) | Primary AI pair programming assistant, used throughout development |
| Claude Opus 4.6 | Model powering the coding assistant |
| Claude Sonnet (`claude-sonnet-4-20250514`) | Runtime LLM for answer generation in the MCP server |
| VS Code | Editor and Git integration |
| Conda | Python environment management |

### Development Workflow

The project was built using a deliberate, incremental workflow designed to maintain understanding and control at every step:

1. **Plan first** — Before writing any code, I created a detailed plan (`PLAN.md`) covering architecture, tech stack, and module breakdown. Claude helped evaluate trade-offs (e.g., ChromaDB vs FAISS, local vs API embeddings) and I made the final decisions.

2. **CLAUDE.md as living source of truth** — A project context file (`CLAUDE.md`) was maintained throughout development. It tracked architecture decisions, current status, eval results, and coding rules. This gave the AI full project context in every session and kept the human and AI aligned.

3. **One function at a time** — Each function was explained by the AI before implementation, reviewed for correctness, then committed separately. This kept commits focused and ensured I understood every line of code in the project.

4. **Iterate on failures** — When something didn't work (ChromaDB build failure, OpenAI billing, Ollama timeouts), the AI proposed alternatives and explained trade-offs. I chose which direction to take. When the eval produced false positives, the AI analyzed all 30 results individually and we iteratively improved the matching logic together.

### Shipping Under Constraints — The Forward-Deployed Mindset

This project hit real-world constraints that required practical problem-solving over ideal solutions:

| Constraint | Problem | Practical Solution |
|---|---|---|
| Windows development | ChromaDB requires C++ build tools | Switched to FAISS (prebuilt wheels, zero setup) |
| No OpenAI billing | Couldn't use OpenAI embedding API | Switched to sentence-transformers (local, free) |
| Local hardware limits | Ollama timed out generating answers | Switched to Claude API (fast, reliable) |
| Eval false positives | 50% word overlap too lenient | Added NLTK stop word filtering + refusal detection |

Each pivot was documented as a trade-off decision, not hidden. The final system uses the best tool for each job given the actual constraints — not the theoretically ideal choice.

### Where I Leaned on AI vs. Overrode It

**Leaned on AI:**
- Architecture planning and trade-off analysis (ChromaDB vs FAISS, embedding model selection)
- Debugging environment-specific issues (Windows build tools, API configuration)
- Writing boilerplate code (FAISS index setup, FastMCP tool decorators, CLI scripts)
- Analyzing eval results — Claude reviewed all 30 Q&A outputs and identified false positives faster than manual inspection

**Overrode or corrected AI:**
- Claude tried to write entire files at once — I enforced one-function-at-a-time commits
- Claude suggested manual stop word lists — I insisted on using NLTK's programmatic solution
- Claude initially proposed committing files one-by-one instead of logical groupings — I restructured the commit strategy
- Claude's metadata example output was fabricated — I ran the code and replaced it with real output

### What I Learned

- **AI is most useful as a collaborator, not an autopilot.** The best results came from having Claude explain its reasoning, then making my own decision. When I let it write too much at once, I lost track of what the code was doing.

- **Incremental development catches problems early.** Building one function at a time meant I could test and verify at each step. The eval false positives were caught because I reviewed every test result individually rather than trusting the summary numbers.

- **Environment issues are the real time sink.** The actual RAG pipeline code was straightforward. The majority of debugging time went to environment-specific issues (Windows compatibility, API key setup, model timeouts). AI was particularly effective here — identifying platform-specific issues and suggesting alternatives I wouldn't have known about.

- **Honest eval > impressive numbers.** The initial eval showed 100% on some categories, which was clearly wrong. Instead of shipping inflated numbers, I invested time in identifying false positives, adding stop word filtering, and reporting accurate results. The final numbers are lower but trustworthy.

### What Didn't Work

- Claude occasionally generated complete file rewrites when only a small change was needed — I had to explicitly enforce the one-function-at-a-time rule.
- The eval matching went through three iterations (strict string match → word overlap → stop-word-filtered overlap with refusal detection) before reaching a reasonable balance. Each version had blind spots.
- Some metadata questions (author names, word counts) score 0% because the eval harness only tests the RAG tool, not the metadata tool. This is a known gap, not a system failure.

### Integrity Note

All modules are fully generic. The AI did not write any question-specific or hardcoded logic. The system prompt, retrieval pipeline, and evaluation harness contain no knowledge of the specific test questions. The system works with any set of PDF documents.
