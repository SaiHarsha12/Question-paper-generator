"""Unit tests for Empirical Evaluation Benchmark Harness.

Course: Natural Language Processing for LLM (24AM7PCLLM)
Institution: B.M.S. College of Engineering (BMSCE)
"""

import unittest
import os
import json
import tempfile
import pandas as pd

from evaluation.run_evaluation import (
    load_test_cases,
    build_evaluation_knowledge_base,
    run_evaluation,
    TEST_CASES_PATH
)


class TestEvaluationHarness(unittest.TestCase):
    """Test suite for evaluation harness, test cases, and metric aggregation."""

    def test_load_test_cases_structure(self):
        """Verify test_cases.json has at least 5 test cases with required academic fields."""
        cases = load_test_cases(TEST_CASES_PATH)
        self.assertGreaterEqual(len(cases), 5, "Must have at least 5 test cases")

        case_types = set(c.get("case_type") for c in cases)
        self.assertIn("table_course_plan", case_types, "Must include a table-style course plan test case")
        self.assertIn("scanned_ocr", case_types, "Must include a scanned OCR syllabus test case")

        difficulties = set(c.get("difficulty") for c in cases)
        self.assertIn("Easy", difficulties)
        self.assertIn("Moderate", difficulties)
        self.assertIn("Hard", difficulties)

        for c in cases:
            self.assertIn("id", c)
            self.assertIn("name", c)
            self.assertIn("unit", c)
            self.assertIn("subtopic", c)
            self.assertIn("target_bloom", c)
            self.assertIn("context", c)
            self.assertGreater(len(c["context"]), 50, "Context must be non-trivial")

    def test_build_evaluation_knowledge_base(self):
        """Verify knowledge base builds and indexes chunks from test case contexts."""
        cases = load_test_cases(TEST_CASES_PATH)
        kb = build_evaluation_knowledge_base(cases[:2])
        self.assertGreater(len(kb.chunks), 0)
        results = kb.retrieve("tokenization vocabulary", k=2)
        self.assertGreater(len(results), 0)

    def test_run_evaluation_mini_pipeline(self):
        """Verify run_evaluation executes, produces CSV, Markdown, and Matplotlib figures."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a 1-case test json for fast integration test
            mini_case = [
                {
                    "id": "TC_MINI",
                    "name": "Attention Mechanism Test",
                    "unit": "UNIT-1",
                    "subtopic": "Self-Attention",
                    "difficulty": "Easy",
                    "target_bloom": "Understand",
                    "question_type": "theory",
                    "case_type": "standard_syllabus",
                    "context": "The self-attention mechanism computes representations by relating different positions of a single sequence."
                }
            ]
            mini_json_path = os.path.join(tmp_dir, "mini_test_cases.json")
            with open(mini_json_path, "w", encoding="utf-8") as f:
                json.dump(mini_case, f)

            csv_out = os.path.join(tmp_dir, "eval_out.csv")

            results = run_evaluation(
                test_cases_path=mini_json_path,
                model_name="llama3",
                output_csv=csv_out,
                generate_plots=True
            )

            self.assertTrue(os.path.exists(csv_out))
            self.assertIn("trials", results)
            self.assertEqual(len(results["trials"]), 4, "Should run 4 configurations for 1 test case")
            self.assertIn("aggregated", results)
            self.assertEqual(len(results["aggregated"]), 4)
            self.assertIn("figure_paths", results)
            self.assertIn("relevance_vs_groundedness", results["figure_paths"])


if __name__ == "__main__":
    unittest.main()

