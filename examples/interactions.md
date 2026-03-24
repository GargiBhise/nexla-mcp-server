# Example Interactions

Sample Q&A interactions from the MCP server's `query_documents` tool, showing questions across different types and documents.

---

## 1. Text-Only Question (PASS)

**Question:** How many categories of aggression were participants asked to classify texts into?

**Answer:** Participants were asked to classify texts into 3 categories: overt aggression, covert aggression, and non-aggression.

**Sources:** W18-4401.pdf, Page 1

**Match:** 6/6 content words (100%)

---

## 2. Table Question (PASS)

**Question:** What loss function achieved the highest average performance according to Table 5?

**Answer:** According to Table 5, the "cloze" loss function achieved the highest average performance with a score of 82.3.

**Sources:** D19-1539.pdf, Page 7

**Match:** 10/12 content words (83%)

---

## 3. Table Question (PASS)

**Question:** Which method produces the longest sentences in terms of word count?

**Answer:** Looking at the statistics provided in the context, the MemNet method produces the longest sentences with a word count of 13.1.

**Sources:** W18-5713.pdf, Page 4

**Match:** 6/7 content words (86%)

---

## 4. Cross-Document Question

**Question:** What techniques do the papers use to handle language-specific challenges in NLP tasks?

**Why this is cross-document:** The retriever pulls chunks from multiple PDFs (e.g., P19-1164.pdf on gender bias in translation, W18-4401.pdf on aggression detection across languages, P19-1598.pdf on knowledge-grounded language models) and the answerer synthesizes a single response citing multiple sources.

---

## 5. Refusal Example (correctly handled)

**Question:** Who is the last author of the paper?

**Expected:** Sameer Singh

**Answer:** I don't know the answer.

**Why:** This is a metadata question that requires knowing author order — the RAG pipeline retrieves text chunks, not structured metadata. The `get_document_metadata` tool would handle this.

---

## 6. list_documents Tool

**Input:** *(no parameters)*

**Output:**
```json
["P19-1598.pdf", "W18-4401.pdf", "P19-1164.pdf", "D19-1539.pdf", "W18-5713.pdf"]
```

---

## 7. get_document_metadata Tool

**Input:** `filename = "W18-5713.pdf"`

**Output:**
```json
{
  "filename": "W18-5713.pdf",
  "title": "Retrieve and Refine:",
  "authors": [
    "Improved Sequence Generation Models For Dialogue",
    "JasonWeston,EmilyDinanandAlexanderH.Miller",
    "FacebookAIResearch"
  ],
  "page_count": 6,
  "word_count": 2370,
  "reference_count": 12
}
```

**Note:** Title and author extraction is best-effort using heuristics on the first page. The title is truncated because the subtitle wraps to the next line, and authors are not cleanly separated due to PDF text extraction limitations. This is a known limitation of text-based metadata parsing.
