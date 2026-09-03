from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


OV_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(OV_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(OV_TEST_ROOT))

from src.core.pdf_text import (
    PDFTextResult,
    _normalize_diagnostics,
    extract_pdf_text,
    summarize_diagnostics,
)


class PDFTextTests(unittest.TestCase):
    def test_pymupdf_extracts_text_from_a_valid_pdf(self):
        import pymupdf

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.pdf"
            with pymupdf.open() as document:
                page = document.new_page()
                page.insert_text((72, 72), "GraphRAG PDF extraction")
                document.save(path)

            result = extract_pdf_text(path)

        self.assertEqual(result.backend, "pymupdf")
        self.assertEqual(result.page_count, 1)
        self.assertIn("GraphRAG PDF extraction", result.text)
        self.assertEqual(result.fallback_reason, "")

    def test_falls_back_to_pypdf_when_pymupdf_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fallback.pdf"
            path.write_bytes(b"%PDF-test")
            with (
                patch(
                    "src.core.pdf_text._extract_with_pymupdf",
                    side_effect=ValueError("broken content stream"),
                ),
                patch(
                    "src.core.pdf_text._extract_with_pypdf",
                    return_value=("fallback text", 3),
                ),
            ):
                result = extract_pdf_text(path)

        self.assertEqual(result.backend, "pypdf")
        self.assertEqual(result.page_count, 3)
        self.assertEqual(result.text, "fallback text")
        self.assertIn("broken content stream", result.fallback_reason)

    def test_falls_back_to_pypdf_when_pymupdf_returns_empty_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.pdf"
            path.write_bytes(b"%PDF-test")
            primary = PDFTextResult(
                text="",
                backend="pymupdf",
                page_count=2,
                diagnostics=("syntax error",),
            )
            with (
                patch(
                    "src.core.pdf_text._extract_with_pymupdf",
                    return_value=primary,
                ),
                patch(
                    "src.core.pdf_text._extract_with_pypdf",
                    return_value=("recovered text", 2),
                ),
            ):
                result = extract_pdf_text(path)

        self.assertEqual(result.backend, "pypdf")
        self.assertEqual(result.text, "recovered text")
        self.assertEqual(result.diagnostics, ("syntax error",))
        self.assertIn("no extractable text", result.fallback_reason)

    def test_reports_both_backend_failures_with_the_pdf_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.pdf"
            path.write_bytes(b"%PDF-test")
            with (
                patch(
                    "src.core.pdf_text._extract_with_pymupdf",
                    side_effect=ValueError("primary failed"),
                ),
                patch(
                    "src.core.pdf_text._extract_with_pypdf",
                    side_effect=ValueError("fallback failed"),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "primary failed.*fallback failed"
                ) as raised:
                    extract_pdf_text(path)

        self.assertIn(str(path), str(raised.exception))

    def test_diagnostics_are_deduplicated_and_bounded(self):
        diagnostics = _normalize_diagnostics(
            "syntax error: rgb\nsyntax   error: rgb\nsyntax error: gray\nthird\nfourth\n"
        )
        self.assertEqual(
            diagnostics,
            ("syntax error: rgb", "syntax error: gray", "third", "fourth"),
        )
        self.assertEqual(
            summarize_diagnostics(diagnostics, limit=2),
            "syntax error: rgb; syntax error: gray; ... and 2 more",
        )


if __name__ == "__main__":
    unittest.main()
