"""Unit tests for Question Paper Quality Assurance and Grounding Subsystem.

Course: Natural Language Processing for LLM (24AM7PCLLM)
Institution: B.M.S. College of Engineering (BMSCE)
"""

import unittest
from modules.quality import (
    relevance_score,
    duplicate_check,
    is_candidate_duplicate,
    groundedness_score,
    format_check,
    unit_coverage,
    quality_report
)


class TestQualitySubsystem(unittest.TestCase):
    """Test suite for quality, groundedness, formatting, and duplicate checks."""

    def test_relevance_score_semantic_alignment(self):
        """Verify high relevance score for context-aligned questions."""
        q = "Explain the self-attention mechanism and dot-product scaling in transformer architectures."
        chunks = [
            {"text": "The transformer model utilizes scaled dot-product self-attention to compute token representations across multiple heads."}
        ]
        score = relevance_score(q, chunks)
        self.assertGreaterEqual(score, 0.6)
        self.assertLessEqual(score, 1.0)

    def test_relevance_score_unrelated_context(self):
        """Verify lower relevance score when question is unrelated to retrieved context."""
        q = "Design a distributed database sharding strategy for high-throughput banking systems."
        chunks = [
            {"text": "Photosynthesis is the biochemical process by which green plants convert sunlight into chemical energy stored in glucose."}
        ]
        score = relevance_score(q, chunks)
        self.assertLess(score, 0.45)

    def test_relevance_score_baseline_fallback(self):
        """Verify empty chunks in baseline mode returns neutral 1.0 without error."""
        q = "Define the bias-variance tradeoff in machine learning."
        score = relevance_score(q, [])
        self.assertEqual(score, 1.0)

    def test_duplicate_check_flags_near_duplicates(self):
        """Verify pairwise duplicate detection flags near-identical questions."""
        q1 = "Explain the mathematical formulation of the scaled dot-product attention mechanism."
        q2 = "Explain the mathematical formulation of scaled dot-product attention mechanisms."
        q3 = "Design an automated customer onboarding pipeline with microservices."
        
        flagged = duplicate_check([q1, q2, q3], threshold=0.85)
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["idx1"], 0)
        self.assertEqual(flagged[0]["idx2"], 1)
        self.assertGreaterEqual(flagged[0]["similarity"], 0.85)

    def test_is_candidate_duplicate(self):
        """Verify is_candidate_duplicate identifies matching existing questions."""
        existing = [
            "Explain the architecture of BERT and its bidirectional self-attention encoders.",
            "Illustrate the process of gradient clipping to mitigate exploding gradients."
        ]
        candidate_dup = "Explain the architecture of BERT using its bidirectional self-attention encoder blocks."
        candidate_unique = "Solve the maximum flow problem using the Ford-Fulkerson algorithm."

        is_dup, sim, idx = is_candidate_duplicate(candidate_dup, existing, threshold=0.85)
        self.assertTrue(is_dup)
        self.assertEqual(idx, 0)
        self.assertGreaterEqual(sim, 0.85)

        is_dup_unique, sim_u, _ = is_candidate_duplicate(candidate_unique, existing, threshold=0.85)
        self.assertFalse(is_dup_unique)
        self.assertLess(sim_u, 0.6)

    def test_groundedness_score_high_overlap(self):
        """Verify high groundedness score when question words originate from context."""
        q = "Explain parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA) on frozen weights."
        context = "Low-Rank Adaptation (LoRA) freezes the pre-trained model weights and injects trainable rank decomposition matrices into layers for parameter-efficient fine-tuning."
        score = groundedness_score(q, context)
        self.assertGreaterEqual(score, 0.75)

    def test_groundedness_score_hallucinated_concepts(self):
        """Verify low groundedness score when question concepts do not exist in context."""
        q = "Evaluate quantum teleportation protocols using superconducting qubit architectures."
        context = "Decision trees partition the input space recursively using information gain or Gini impurity."
        score = groundedness_score(q, context)
        self.assertLessEqual(score, 0.25)

    def test_format_check_valid_question(self):
        """Verify compliant exam question passes format verification."""
        q = "Explain the trade-offs between dense retrieval with FAISS and sparse lexical retrieval with BM25."
        res = format_check(q)
        self.assertTrue(res["valid"])
        self.assertTrue(res["starts_with_verb"])
        self.assertTrue(res["no_intro"])
        self.assertTrue(res["length_valid"])
        self.assertTrue(res["no_marks"])
        self.assertEqual(len(res["issues"]), 0)

    def test_format_check_rejections(self):
        """Verify rejection of conversational intros, marks annotations, and non-verb stems."""
        # Intro chatter
        res_intro = format_check("Here is your question: Explain the attention mechanism in detail.")
        self.assertFalse(res_intro["valid"])
        self.assertFalse(res_intro["no_intro"])

        # Marks annotations
        res_marks = format_check("Explain the self-attention mechanism in transformer models (10 marks).")
        self.assertFalse(res_marks["valid"])
        self.assertFalse(res_marks["no_marks"])

        # Too short (< 8 words)
        res_short = format_check("Define deep learning.")
        self.assertFalse(res_short["valid"])
        self.assertFalse(res_short["length_valid"])

        # Non-verb start
        res_noverb = format_check("The transformer model is widely used in NLP tasks because of attention.")
        self.assertFalse(res_noverb["valid"])
        self.assertFalse(res_noverb["starts_with_verb"])

    def test_unit_coverage_validation(self):
        """Verify unit coverage correctly detects complete and incomplete papers."""
        complete_paper = [
            {"unit": "UNIT-1", "topic": "T1"}, {"unit": "UNIT-1", "topic": "T2"},
            {"unit": "UNIT-1", "topic": "T3"}, {"unit": "UNIT-1", "topic": "T4"},
            {"unit": "UNIT-2", "topic": "T5"}, {"unit": "UNIT-2", "topic": "T6"},
            {"unit": "UNIT-2", "topic": "T7"}, {"unit": "UNIT-2", "topic": "T8"}
        ]
        rep = unit_coverage(complete_paper, expected_units=["UNIT-1", "UNIT-2"], expected_per_unit=4)
        self.assertTrue(rep["complete"])
        self.assertEqual(rep["units_covered"], 2)

        incomplete_paper = [
            {"unit": "UNIT-1", "topic": "T1"}, {"unit": "UNIT-1", "topic": "T2"}
        ]
        rep_inc = unit_coverage(incomplete_paper, expected_units=["UNIT-1", "UNIT-2"], expected_per_unit=4)
        self.assertFalse(rep_inc["complete"])
        self.assertGreater(len(rep_inc["issues"]), 0)

    def test_quality_report_aggregation(self):
        """Verify complete quality report aggregates metrics accurately."""
        paper = [
            {
                "qno": 1,
                "unit": "UNIT-1",
                "topic": "Attention",
                "question": "Explain the self-attention mechanism and scaled dot-product formulation.",
                "retrieved_chunks": ["The attention mechanism computes scaled dot-product attention over sequence representations."]
            },
            {
                "qno": 2,
                "unit": "UNIT-1",
                "topic": "LoRA",
                "question": "Evaluate the computational trade-offs of parameter-efficient fine-tuning with LoRA.",
                "retrieved_chunks": ["LoRA is a parameter-efficient fine-tuning method analyzing computational trade-offs and trainable weights."]
            }
        ]
        report = quality_report(paper)
        self.assertIn("summary", report)
        self.assertEqual(report["summary"]["total_questions"], 2)
        self.assertGreaterEqual(report["summary"]["avg_groundedness"], 0.6)
        self.assertEqual(report["summary"]["duplicate_count"], 0)
        self.assertEqual(len(report["per_question"]), 2)


if __name__ == "__main__":
    unittest.main()
