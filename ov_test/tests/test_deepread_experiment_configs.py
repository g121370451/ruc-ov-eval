import os
import unittest

import yaml


class DeepReadExperimentConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_dir = os.path.join(
            os.path.dirname(__file__), "..", "config_deepread_global"
        )
        cls.configs = {}
        for name in (
            "financebench_graph_keyword",
            "financebench_no_community",
            "financebench_original",
        ):
            path = os.path.join(config_dir, f"{name}.yaml")
            with open(path, "r", encoding="utf-8") as handle:
                cls.configs[name] = yaml.safe_load(handle)

    def test_experiments_use_separate_storage_and_output_paths(self):
        for key in ("doc_output_dir", "vector_store", "output_dir"):
            values = [config["paths"][key] for config in self.configs.values()]
            self.assertEqual(len(values), len(set(values)), key)

    def test_graph_and_compatibility_switches_are_explicit(self):
        graph = self.configs["financebench_graph_keyword"]["store"]
        no_community = self.configs["financebench_no_community"]["store"]
        original = self.configs["financebench_original"]["store"]

        self.assertTrue(graph["enable_document_graph"])
        self.assertFalse(no_community["enable_document_graph"])
        self.assertFalse(original["enable_document_graph"])
        self.assertTrue(no_community["enable_read_label_dedup"])
        self.assertFalse(original["enable_read_label_dedup"])
        self.assertFalse(no_community["enable_vector"])
        self.assertTrue(original["enable_vector"])


if __name__ == "__main__":
    unittest.main()
