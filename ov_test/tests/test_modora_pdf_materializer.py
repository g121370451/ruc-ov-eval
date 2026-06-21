import sys
import tempfile
import unittest
from pathlib import Path

import fitz


REPO_ROOT = Path(__file__).resolve().parents[2]
OV_TEST_ROOT = REPO_ROOT / "ov_test"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(OV_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(OV_TEST_ROOT))

from src.adapters.base import StandardDoc
from src.core.modora_store import ModoraStoreWrapper


class ModoraPdfMaterializerTest(unittest.TestCase):
    def _wrapper(self, root: Path) -> ModoraStoreWrapper:
        return ModoraStoreWrapper(
            store_path=str(root / "store"),
            modora_config={
                "docs_dir": str(root / "docs"),
                "cache_dir": str(root / "cache"),
                "ingest_mode": "none",
                "preload_library": False,
            },
        )

    def test_markdown_is_materialized_to_readable_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            md_path = root / "source.md"
            md_path.write_text("# Title\n\nThis is a materialized document.", encoding="utf-8")

            wrapper = self._wrapper(root)
            samples, stats = wrapper._materialize_pdf_documents(
                [StandardDoc(sample_id="sample-1", doc_paths=[str(md_path)])]
            )

            self.assertEqual(stats["converted_text"], 1)
            pdf_path = Path(samples[0].doc_paths[0])
            self.assertTrue(pdf_path.exists())

            doc = fitz.open(pdf_path)
            try:
                text = "\n".join(page.get_text("text") for page in doc)
            finally:
                doc.close()
            self.assertIn("materialized document", text)

    def test_pdf_input_is_copied_and_duplicate_references_share_one_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_pdf = root / "same_name.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Original PDF")
            doc.save(source_pdf)
            doc.close()

            wrapper = self._wrapper(root)
            samples, stats = wrapper._materialize_pdf_documents(
                [
                    StandardDoc(sample_id="a", doc_paths=[str(source_pdf)]),
                    StandardDoc(sample_id="b", doc_paths=[str(source_pdf)]),
                ]
            )

            self.assertEqual(stats["unique_sources"], 1)
            self.assertEqual(stats["copied_pdf"], 1)
            self.assertEqual(samples[0].doc_paths, samples[1].doc_paths)
            self.assertTrue(Path(samples[0].doc_paths[0]).exists())

    def test_duplicate_stems_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left"
            right = root / "right"
            left.mkdir()
            right.mkdir()
            left_doc = left / "doc.md"
            right_doc = right / "doc.md"
            left_doc.write_text("left document", encoding="utf-8")
            right_doc.write_text("right document", encoding="utf-8")

            wrapper = self._wrapper(root)
            samples, stats = wrapper._materialize_pdf_documents(
                [StandardDoc(sample_id="sample", doc_paths=[str(left_doc), str(right_doc)])]
            )

            self.assertEqual(stats["converted_text"], 2)
            self.assertEqual(len(samples[0].doc_paths), 2)
            self.assertNotEqual(samples[0].doc_paths[0], samples[0].doc_paths[1])

    def test_unsupported_extension_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_path = root / "data.json"
            bad_path.write_text("{}", encoding="utf-8")

            wrapper = self._wrapper(root)
            with self.assertRaisesRegex(ValueError, "Unsupported source document extension"):
                wrapper._materialize_pdf_documents(
                    [StandardDoc(sample_id="sample", doc_paths=[str(bad_path)])]
                )

    def test_clear_removes_configured_vector_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs"
            cache = root / "cache"
            chroma = root / "store_index"
            docs.mkdir()
            cache.mkdir()
            chroma.mkdir()
            (chroma / "chroma.sqlite3").write_text("index", encoding="utf-8")

            wrapper = ModoraStoreWrapper(
                store_path=str(chroma),
                modora_config={
                    "docs_dir": str(docs),
                    "cache_dir": str(cache),
                    "chroma_persist_path": str(chroma),
                    "delete_vector_index": True,
                    "ingest_mode": "none",
                    "preload_library": False,
                },
            )

            wrapper.clear()

            self.assertFalse(chroma.exists())


if __name__ == "__main__":
    unittest.main()
