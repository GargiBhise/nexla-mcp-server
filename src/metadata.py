import re
import pdfplumber


def extract_metadata(pdf_path: str, filename: str) -> dict:
    """
    Extract document-level metadata from a PDF file.
    Returns title, authors, page count, word count, and reference count.
    """
    # Default values in case extraction fails
    metadata = {
        "filename": filename,
        "title": "",
        "authors": [],
        "page_count": 0,
        "word_count": 0,
        "reference_count": 0,
    }

    with pdfplumber.open(pdf_path) as pdf:
        # Total number of pages
        metadata["page_count"] = len(pdf.pages)

        # Concatenate text from all pages
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text() or ""  # fallback to empty string if page has no text
            full_text += text + "\n"

        # Count total words across the document
        metadata["word_count"] = len(full_text.split())

        # Delegate to helper functions for structured fields
        metadata["title"] = _extract_title(pdf)
        metadata["authors"] = _extract_authors(pdf)
        metadata["reference_count"] = _count_references(full_text)

    return metadata


def _clean_text(text: str) -> str:
    """Remove unicode superscripts and special markers from extracted PDF text."""
    # Strip common unicode superscript characters (*, †, §, ‡) and digit superscripts
    cleaned = re.sub(r'[\u2217\u2020\u2021\u00a7\u00b9\u00b2\u00b3]', '', text)
    # Remove standalone superscript-style digits that pdfplumber extracts (e.g. "1" after a name)
    cleaned = re.sub(r'(?<=[a-zA-Z])\d{1,2}(?=\s|,|$)', '', cleaned)
    return cleaned.strip()


def _extract_title(pdf) -> str:
    """Extract title from the first page (first non-empty line)."""
    first_page = pdf.pages[0]
    text = first_page.extract_text() or ""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return _clean_text(lines[0]) if lines else ""


def _extract_authors(pdf) -> list[str]:
    """Extract authors from the first page (lines between title and abstract)."""
    first_page = pdf.pages[0]
    text = first_page.extract_text() or ""
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Skip lines that are affiliations, locations, or section headers
    skip_keywords = ["abstract", "introduction", "university", "department",
                     "institute", "@", "research", "menlo", "seattle", "CA", "WA"]

    authors = []
    # Look at lines 2-6 (after title) for author names
    for line in lines[1:6]:
        lower = line.lower()
        # Stop at section headers
        if "abstract" in lower or "introduction" in lower:
            break
        # Skip affiliations and locations
        if any(kw.lower() in lower for kw in skip_keywords):
            continue
        # Skip very short lines (stray numbers or symbols)
        if len(line) < 5:
            continue

        # Clean unicode artifacts and add as author line
        cleaned = _clean_text(line)
        if cleaned:
            authors.append(cleaned)

    return authors


def _count_references(full_text: str) -> int:
    """Count the number of references in the references section."""
    # Find where the references section starts
    ref_match = re.search(r'\bReferences\b', full_text, re.IGNORECASE)
    if not ref_match:
        return 0

    ref_section = full_text[ref_match.start():]

    # Count bracketed references e.g. [1], [2], [3]
    bracketed = re.findall(r'^\[\d+\]', ref_section, re.MULTILINE)
    if bracketed:
        return len(bracketed)

    # Fallback: count author-year style references e.g. "Smith, J"
    author_year = re.findall(r'^[A-Z][a-z]+,?\s+[A-Z]', ref_section, re.MULTILINE)
    return len(author_year)
