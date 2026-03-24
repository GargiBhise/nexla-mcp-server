import re
import pdfplumber


def extract_metadata(pdf_path: str, filename: str) -> dict:
    """
    Extract document-level metadata from a PDF file.
    Returns title, authors, page count, word count, and reference count.
    """
    metadata = {
        "filename": filename,
        "title": "",
        "authors": [],
        "page_count": 0,
        "word_count": 0,
        "reference_count": 0,
    }

    with pdfplumber.open(pdf_path) as pdf:
        metadata["page_count"] = len(pdf.pages)

        full_text = ""
        for page in pdf.pages:
            text = page.extract_text() or ""
            full_text += text + "\n"

        metadata["word_count"] = len(full_text.split())
        metadata["title"] = _extract_title(pdf)
        metadata["authors"] = _extract_authors(pdf)
        metadata["reference_count"] = _count_references(full_text)

    return metadata


def _extract_title(pdf) -> str:
    """Extract title from the first page (first non-empty line)."""
    pass


def _extract_authors(pdf) -> list[str]:
    """Extract authors from the first page (lines between title and abstract)."""
    pass


def _count_references(full_text: str) -> int:
    """Count the number of references in the references section."""
    pass
