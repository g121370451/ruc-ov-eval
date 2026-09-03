"""Robust PDF text extraction for document-indexing backends."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path


_PYMUPDF_DIAGNOSTICS_LOCK = threading.Lock()


@dataclass(frozen=True)
class PDFTextResult:
    """Text and provenance returned by one PDF extraction attempt."""

    text: str
    backend: str
    page_count: int
    diagnostics: tuple[str, ...] = ()
    fallback_reason: str = ""


class _PyMuPDFExtractionError(RuntimeError):
    def __init__(self, cause: Exception, diagnostics: tuple[str, ...]) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause
        self.diagnostics = diagnostics


def extract_pdf_text(path: str | Path) -> PDFTextResult:
    """Extract a PDF with PyMuPDF, falling back to pypdf on hard failure.

    MuPDF emits recoverable parser diagnostics through a process-global message
    channel.  They are captured here instead of being printed without a filename.
    A recoverable diagnostic does not by itself trigger fallback: if PyMuPDF still
    returns text, that text remains the primary result.  Exceptions and empty text
    do trigger a full-document pypdf retry.
    """

    pdf_path = Path(path).resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    primary_diagnostics: tuple[str, ...] = ()
    try:
        primary = _extract_with_pymupdf(pdf_path)
        primary_diagnostics = primary.diagnostics
        if primary.text.strip():
            return primary
        primary_failure = "PyMuPDF returned no extractable text"
    except _PyMuPDFExtractionError as exc:
        primary_diagnostics = exc.diagnostics
        primary_failure = f"PyMuPDF {exc}"
    except Exception as exc:  # pypdf gets one chance to recover the document
        primary_failure = f"PyMuPDF {type(exc).__name__}: {exc}"

    try:
        fallback_text, page_count = _extract_with_pypdf(pdf_path)
    except Exception as exc:
        raise RuntimeError(
            f"PDF text extraction failed for {pdf_path}: {primary_failure}; "
            f"pypdf {type(exc).__name__}: {exc}"
        ) from exc

    if not fallback_text.strip():
        raise RuntimeError(
            f"PDF text extraction produced no text for {pdf_path}: "
            f"{primary_failure}; pypdf also returned empty text. "
            "The document may be scanned and require OCR."
        )

    return PDFTextResult(
        text=fallback_text,
        backend="pypdf",
        page_count=page_count,
        diagnostics=primary_diagnostics,
        fallback_reason=primary_failure,
    )


def _extract_with_pymupdf(path: Path) -> PDFTextResult:
    import pymupdf

    # These switches and the diagnostics buffer are process-global in PyMuPDF.
    # Serialize extraction so parallel stores cannot consume each other's messages.
    with _PYMUPDF_DIAGNOSTICS_LOCK:
        display_errors = pymupdf.TOOLS.mupdf_display_errors()
        display_warnings = pymupdf.TOOLS.mupdf_display_warnings()
        pymupdf.TOOLS.reset_mupdf_warnings()
        pymupdf.TOOLS.mupdf_display_errors(False)
        pymupdf.TOOLS.mupdf_display_warnings(False)

        diagnostics = ""
        pages: list[str] = []
        page_count = 0
        extraction_error: Exception | None = None
        try:
            with pymupdf.open(path) as document:
                page_count = document.page_count
                pages = [
                    page.get_text(
                        "text",
                        flags=pymupdf.TEXTFLAGS_TEXT,
                        sort=True,
                    ).strip()
                    for page in document
                ]
        except Exception as exc:  # preserve diagnostics before trying pypdf
            extraction_error = exc
        finally:
            diagnostics = pymupdf.TOOLS.mupdf_warnings(reset=1) or ""
            pymupdf.TOOLS.mupdf_display_errors(display_errors)
            pymupdf.TOOLS.mupdf_display_warnings(display_warnings)

    normalized_diagnostics = _normalize_diagnostics(diagnostics)
    if extraction_error is not None:
        raise _PyMuPDFExtractionError(
            extraction_error, normalized_diagnostics
        ) from extraction_error

    return PDFTextResult(
        text="\n\n".join(page for page in pages if page).strip(),
        backend="pymupdf",
        page_count=page_count,
        diagnostics=normalized_diagnostics,
    )


def _extract_with_pypdf(path: Path) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(str(path), strict=False)
    if reader.is_encrypted:
        decrypt_result = reader.decrypt("")
        if not decrypt_result:
            raise ValueError("encrypted PDF requires a password")
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(page for page in pages if page).strip(), len(reader.pages)


def _normalize_diagnostics(raw: str) -> tuple[str, ...]:
    unique: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        normalized = " ".join(line.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return tuple(unique)


def summarize_diagnostics(diagnostics: tuple[str, ...], limit: int = 5) -> str:
    """Return a bounded, readable diagnostic summary for logs."""

    shown = list(diagnostics[:limit])
    remaining = len(diagnostics) - len(shown)
    summary = "; ".join(shown)
    if remaining > 0:
        summary += f"; ... and {remaining} more"
    return summary
