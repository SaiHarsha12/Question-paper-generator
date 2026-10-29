"""Unit and Integration Tests for Refactored RAG Generation Engine.

Course: Natural Language Processing for LLM (24AM7PCLLM)
Institution: B.M.S. College of Engineering (BMSCE)
Department: Artificial Intelligence & Machine Learning
"""

import unittest
from typing import Dict, Any, List

from modules.retriever import KnowledgeBase
from modules.rag_engine import (
    generate_exam_question,
    clean_question_text,
    _select_action_verb,
    BLOOM_VERBS,
)


class TestRAGEngine(unittest.TestCase):
    """Test suite for RAG Generation Pipeline."""

    @classmethod
    def setUpClass(cls):
        """Prepare mock KnowledgeBase for testing RAG context injection."""
        cls.kb = KnowledgeBase()
        mock_chunks = [
            {
                "chunk_id": "test_chunk_1",
                "text": "The Bellman-Ford algorithm computes shortest paths from a single source vertex to all other vertices in a weighted digraph, and handles negative weight cycles.",
                "source": "CNS.pdf",
                "page": 1
            },
            {
                "chunk_id": "test_chunk_2",
                "text": "Public-key cryptography utilizes key pairs (public and private). RSA encryption relies on the computational difficulty of factoring the product of two large prime numbers.",
                "source": "CNS.pdf",
                "page": 2
            }
        ]
        cls.kb.build_from_chunks(mock_chunks)

    def test_return_dict_structure(self):
        """Verify generate_exam_question returns the required evaluation dictionary."""
        result = generate_exam_question(
            subtopic="Bellman-Ford routing",
            bloom_level="understand",
            question_type="theory",
            difficulty="Moderate",
            kb=self.kb,
            mode="rag",
            prompt_style="few_shot"
        )
        self.assertIsInstance(result, dict)
        expected_keys = {"question", "retrieved_chunks", "prompt_used", "latency_s", "mode", "prompt_style"}
        for k in expected_keys:
            self.assertIn(k, result)

        self.assertEqual(result["mode"], "rag")
        self.assertEqual(result["prompt_style"], "few_shot")
        self.assertGreater(result["latency_s"], 0.0)
        self.assertGreater(len(result["question"]), 10)

    def test_baseline_mode_no_retrieval(self):
        """Verify baseline mode does not retrieve context chunks."""
        result = generate_exam_question(
            subtopic="TCP vs UDP",
            bloom_level="analyze",
            question_type="theory",
            difficulty="Easy",
            kb=self.kb,
            mode="baseline",
            prompt_style="zero_shot"
        )
        self.assertEqual(result["mode"], "baseline")
        self.assertEqual(result["retrieved_chunks"], [])
        self.assertNotIn("SOURCE CONTEXT", result["prompt_used"])

    def test_rag_mode_context_injection(self):
        """Verify RAG mode retrieves chunks and injects them into the prompt."""
        result = generate_exam_question(
            subtopic="RSA public key factoring",
            bloom_level="apply",
            question_type="numerical",
            difficulty="Moderate",
            kb=self.kb,
            mode="rag",
            prompt_style="zero_shot"
        )
        self.assertEqual(result["mode"], "rag")
        self.assertGreater(len(result["retrieved_chunks"]), 0)
        self.assertIn("SOURCE CONTEXT", result["prompt_used"])
        self.assertIn("factoring the product of two large prime numbers", result["prompt_used"])

    def test_few_shot_prompt_style(self):
        """Verify few_shot prompt style injects Bloom exemplars."""
        result = generate_exam_question(
            subtopic="Network Architecture",
            bloom_level="create",
            question_type="theory",
            difficulty="Hard",
            kb=None,
            mode="baseline",
            prompt_style="few_shot"
        )
        self.assertIn("EXEMPLAR QUESTIONS", result["prompt_used"])
        self.assertEqual(result["prompt_style"], "few_shot")

    def test_bloom_action_verbs(self):
        """Verify appropriate action verbs are selected across all 6 Bloom levels."""
        for level in ["remember", "understand", "apply", "analyze", "evaluate", "create"]:
            verb = _select_action_verb(level)
            self.assertIn(verb, BLOOM_VERBS[level])

    def test_clean_question_text(self):
        """Verify question sanitation strips conversational artifacts and bold markdown."""
        raw = "Here is your question: **Design** an end-to-end network architecture. (10 marks) [Provide diagram]"
        cleaned = clean_question_text(raw)
        self.assertFalse(cleaned.lower().startswith("here is"))
        self.assertNotIn("**", cleaned)
        self.assertNotIn("10 marks", cleaned)
        self.assertNotIn("[Provide diagram]", cleaned)
        self.assertTrue(cleaned.startswith("Design"))


if __name__ == "__main__":
    unittest.main()

