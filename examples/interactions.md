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

**Answer:** Several techniques are used to handle language-specific challenges in NLP tasks:

**Cross-lingual Word Representations:** The forest and trees system uses pre-trained English and Hindi fasttext word embeddings combined with pre-computed SVD matrices that align Hindi and English vectors into a single space, enabling the same model to be used for both languages.

**Language-Independent Representations:** UAEMex-UAPT1 uses word model representations that are language independent, allowing the same approach to be used for both English and Hindi tasks.

**Multilingual Word Vectors:** One approach involves learning word vectors for 157 languages to handle multiple language processing tasks.

**Contextualized Representations:** Deep contextualized word representations are used, with ELMo introducing a novel mechanism to learn combinations of different layers in language models.

**Data Augmentation:** External datasets are used to augment training data, such as incorporating external toxicity datasets for aggression identification tasks.

**Sources:**
- D19-1539.pdf, Page 9
- D19-1539.pdf, Page 2
- P19-1598.pdf, Page 9
- W18-4401.pdf, Page 6
- W18-5713.pdf, Page 1

**Why this is cross-document:** The retriever pulled chunks from 4 different PDFs and the answerer synthesized a single response citing all sources.

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

**Input:** `filename = "D19-1539.pdf"`

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

**Note:** Title and author extraction is best-effort using heuristics on PDF first-page text. Author names may not separate cleanly depending on PDF layout.
