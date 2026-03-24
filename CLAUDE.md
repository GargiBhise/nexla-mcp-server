# CLAUDE.md — Project Context for nexla-mcp-server

Source of truth for Claude Code. Read this before doing anything.

---

## 1. Objective and Goal

Build a working **MCP server** that ingests 5 PDF documents, indexes them, and exposes Q&A tools to any MCP-compatible AI agent. Returns grounded answers with source attribution.

**Assignment:** Nexla Forward Deployed Engineer – AI take-home.

| Criterion | Weight |
|---|---|
| Vibe Coding Setup | 40% |
| Code Quality & Architecture | 35% |
| MCP Protocol Understanding | 25% |

---

## 2. Requirements

- Parse and index all 5 PDFs at startup
- `query_documents`: natural language Q&A with source attribution (filename + page)
- `list_documents`: return indexed PDF filenames
- `get_document_metadata`: return title, authors, page/word/ref counts
- Cross-document queries → single synthesized answer
- MCP-compliant, stdio transport, runnable locally
- Evaluation harness using JSONL ground-truth Q&A pairs

---

## 3. Data

5 ACL Anthology NLP papers + 5 JSONL eval files (29 Q&A pairs total). PDFs are machine-readable. JSONL used for evaluation only.

Question types: `text-only`, `multimodal-t` (tables), `multimodal-f` (figures — known limitation), `meta-data`.

---

## 4. Tech Stack

| Component | Choice |
|---|---|
| MCP Framework | FastMCP |
| PDF Parsing | pdfplumber |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (local) |
| Vector Store | FAISS `faiss-cpu` (in-memory) |
| LLM | Claude via Anthropic API (`claude-sonnet-4-20250514`, configurable via `ANTHROPIC_MODEL` in .env) |
| Transport | stdio |

---

## 5. Trade-offs

| Decision | Choice | Reason |
|---|---|---|
| FAISS not ChromaDB | ChromaDB needs C++ build tools on Windows |
| Rebuild at startup | 5 small PDFs, seconds to re-embed |
| sentence-transformers not OpenAI | Local, zero cost, no API key for embeddings |
| Claude API for LLM | Tried Ollama (timeout issues on local hardware), Claude gives better quality |
| stdio transport | Local-only, Claude Desktop default |
| Eval: 50% word overlap | Strict string match was too rigid; word overlap is more forgiving |

---

## 6. Architecture

```
DATA (5 PDFs + 5 JSONL eval-only)
        │
        ▼
INGESTION (ingest.py + metadata.py)
  pdfplumber → chunk (1500 chars, 200 overlap) → sentence-transformers → FAISS (138 vectors)
        │
        ▼
STORAGE: FAISS index + chunks list + metadata dict (all in-memory)
        │
        ▼
RETRIEVAL (retriever.py)
  question → embed → FAISS top-5 → chunks with metadata
        │
        ▼
ANSWER (answerer.py)
  chunks + question → Claude API → synthesized answer + sources
        │
        ▼
MCP SERVER (server.py — FastMCP + stdio)
  Tool 1: query_documents(question) → {answer, sources}
  Tool 2: list_documents() → [filenames]
  Tool 3: get_document_metadata(filename) → {title, authors, counts}
        │
        ▼
MCP CLIENT (Claude Desktop)
        │
        ▼
EVALUATION (eval/eval.py) — offline, reads JSONL, reports accuracy by type
```

---

## 7. File Structure & Status

```
src/metadata.py      ✅ title, authors, page/word/ref counts
src/ingest.py        ✅ parse, chunk, embed, FAISS index
src/retriever.py     ✅ FAISS similarity search
src/answerer.py      ✅ Claude API with RAG prompt (model from .env)
src/server.py        ✅ FastMCP + 3 tools + startup ingestion
eval/eval.py         ✅ accuracy by question type (50% word overlap match)
.env.example         ✅ ANTHROPIC_API_KEY + ANTHROPIC_MODEL
requirements.txt     ✅ fastmcp, pdfplumber, sentence-transformers, faiss-cpu, anthropic, python-dotenv
```

**Remaining:**
- [x] Run full 30-question eval and record results
- [x] Capture 3+ example interactions in examples/interactions.md
- [x] Write README.md (setup, architecture, tools, limitations, vibe coding)
- [ ] Commit and push all changes

---

## 8. Evaluation Results (30 questions)

```
text-only:     2/6  (33.3%)
multimodal-t:  6/14 (42.9%)  — ~1 false positive from wrong-entity match
multimodal-f:  1/2  (50.0%)
meta-data:     0/8  (0.0%)
```

**Eval improvements made:**
- NLTK stop word filtering to remove common words before comparison
- Refusal detection: LLM instructed to say "I don't know the answer" → auto-FAIL
- Word-to-word set matching instead of substring search

**Why meta-data scores 0%:** These questions (author names, page counts, word counts) need the `get_document_metadata` tool, not RAG. The eval harness only tests `query_documents`.

**Why multimodal-f is limited:** Figure-based questions require image understanding; text-only RAG cannot extract figure content.

---

## 9. Known Limitations

- `multimodal-f` (figure questions) unanswerable by text RAG
- Index rebuilt at startup (persistence is production path)
- Stateless — no conversation memory
- Table parsing best-effort via pdfplumber
- Eval word overlap can miss correct paraphrases or pass wrong-entity matches

---

## 10. Git Commit History (30 commits)

No prefix format: `<action> <what> <context>`

Latest: `switch answer generation from Anthropic API to local Ollama`
Note: answerer.py has since been switched back to Claude API (uncommitted)

**Uncommitted changes:**
- answerer.py → Claude API with model from .env
- requirements.txt → anthropic + python-dotenv instead of requests
- .env.example → ANTHROPIC_API_KEY + ANTHROPIC_MODEL
- eval/eval.py → word overlap matching + debug output
- JSONL files → full 29 Q&A pairs restored

---

## 11. Coding Rules

- One function at a time
- Each function gets its own commit
- Short inline comments
- Explain before implementing
- Commit messages: descriptive, professional, no prefixes
- Do not jump ahead without being asked

---

## 12. How to Run

```bash
conda activate
cd D:/nexla-mcp-server
pip install -r requirements.txt
python -m src.server      # start MCP server
python -m eval.eval       # run evaluation
```

Requires: `ANTHROPIC_API_KEY` in `.env`
