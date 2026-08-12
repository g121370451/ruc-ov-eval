import json
import os
import tempfile
import unittest

from src.adapters.locomo_adapter import LocomoAdapter


class LocomoAdapterPathTest(unittest.TestCase):
    def test_directory_resolves_locomo10_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "locomo10.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)

            adapter = LocomoAdapter(temp_dir)

        self.assertEqual(adapter.raw_file_path, path)

    def test_directory_falls_back_to_capitalized_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "Locomo.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)

            adapter = LocomoAdapter(temp_dir)

        self.assertEqual(adapter.raw_file_path, path)


if __name__ == "__main__":
    unittest.main()
