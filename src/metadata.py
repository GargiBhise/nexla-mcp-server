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


def _extract_title(pdf) -> str:
    """Extract title from the first page (first non-empty line)."""
    first_page = pdf.pages[0]
    text = first_page.extract_text() or ""
    # Split into lines and return the first non-empty one as the title
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines[0] if lines else ""


def _extract_authors(pdf) -> list[str]:
    """Extract authors from the first page (lines between title and abstract)."""
    pass


def _count_references(full_text: str) -> int:
    """Count the number of references in the references section."""
    pass
