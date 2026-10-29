"""Unit tests for Course Outcome (CO) and Programme Outcome (PO) Mapper.

Course: Natural Language Processing for LLM (24AM7PCLLM)
Institution: B.M.S. College of Engineering (BMSCE)
"""

import unittest
import os
import json
from modules.co_mapper import (
    load_course_config,
    save_course_config,
    COMapper,
    map_question_to_co,
    DEFAULT_CONFIG
)


class TestCOMapper(unittest.TestCase):
    """Test suite for semantic CO and PO mapping."""

    @classmethod
    def setUpClass(cls):
        """Initialize mapper once to optimize test execution."""
        cls.mapper = COMapper()

    def test_load_course_config_structure(self):
        """Verify course configuration contains required 24AM7PCLLM attributes."""
        cfg = load_course_config("data/course_config.json")
        self.assertEqual(cfg.get("course_code"), "24AM7PCLLM")
        self.assertEqual(cfg.get("course_name"), "Natural Language Processing for LLM")
        self.assertEqual(cfg.get("semester"), "VII")
        self.assertIn("cos", cfg)
        self.assertIn("CO1", cfg["cos"])
        self.assertIn("CO2", cfg["cos"])
        self.assertIn("CO3", cfg["cos"])
        self.assertIn("CO4", cfg["cos"])
        self.assertIn("co_po_matrix", cfg)

    def test_co_po_matrix_mappings(self):
        """Verify the CO to PO/PSO matrix values defined in requirements."""
        matrix = self.mapper.co_po_matrix
        # CO1: PO1=3, PSO3=3
        self.assertEqual(matrix["CO1"]["PO1"], 3)
        self.assertEqual(matrix["CO1"]["PSO3"], 3)
        # CO2: PO2=3, PSO3=3
        self.assertEqual(matrix["CO2"]["PO2"], 3)
        self.assertEqual(matrix["CO2"]["PSO3"], 3)
        # CO3: PO3=2, PO4=3, PSO3=3
        self.assertEqual(matrix["CO3"]["PO3"], 2)
        self.assertEqual(matrix["CO3"]["PO4"], 3)
        self.assertEqual(matrix["CO3"]["PSO3"], 3)
        # CO4: PO3=3, PO4=3, PO5=3, PO8=2, PO9=2, PSO3=3
        self.assertEqual(matrix["CO4"]["PO3"], 3)
        self.assertEqual(matrix["CO4"]["PO4"], 3)
        self.assertEqual(matrix["CO4"]["PO5"], 3)
        self.assertEqual(matrix["CO4"]["PO8"], 2)
        self.assertEqual(matrix["CO4"]["PO9"], 2)
        self.assertEqual(matrix["CO4"]["PSO3"], 3)

    def test_get_pos_for_co(self):
        """Verify PO retrieval for COs both with and without PSOs."""
        # For PDF PO column (include_pso=False)
        pos_co1 = self.mapper.get_pos_for_co("CO1", include_pso=False)
        self.assertEqual(pos_co1, ["PO1"])

        pos_co3 = self.mapper.get_pos_for_co("CO3", include_pso=False)
        self.assertEqual(pos_co3, ["PO3", "PO4"])

        pos_co4 = self.mapper.get_pos_for_co("CO4", include_pso=False)
        self.assertIn("PO3", pos_co4)
        self.assertIn("PO4", pos_co4)
        self.assertIn("PO5", pos_co4)

        # With PSOs (include_pso=True)
        pos_co3_with_pso = self.mapper.get_pos_for_co("CO3", include_pso=True)
        self.assertEqual(pos_co3_with_pso, ["PO3", "PO4", "PSO3"])

    def test_map_question_to_co3_evaluate(self):
        """Verify an LLM adaptation / evaluation question maps to CO3."""
        q = "Evaluate the trade-offs between LoRA and full parameter fine-tuning on perplexity and memory usage."
        co, sim, pos = self.mapper.map_question(q, bloom_level="Evaluate")
        self.assertEqual(co, "CO3")
        self.assertGreater(sim, 0.3)
        self.assertEqual(pos, ["PO3", "PO4"])

    def test_map_question_to_co4_create(self):
        """Verify a system design / application question maps to CO4."""
        q = "Design and demonstrate a real-world multi-agent LLM application for customer support using team workflows."
        co, sim, pos = self.mapper.map_question(q, bloom_level="Create")
        self.assertEqual(co, "CO4")
        self.assertGreater(sim, 0.3)
        self.assertIn("PO3", pos)
        self.assertIn("PO4", pos)

    def test_map_question_to_co2_analyze(self):
        """Verify a linguistic pattern / computational approach question maps to CO2."""
        q = "Analyze how self-attention heads identify syntactic and contextual relationships in ambiguous sentences."
        co, sim, pos = self.mapper.map_question(q, bloom_level="Analyze")
        self.assertEqual(co, "CO2")
        self.assertGreater(sim, 0.3)
        self.assertEqual(pos, ["PO2"])

    def test_map_question_to_co1_apply(self):
        """Verify an input processing / tokenization question maps to CO1."""
        q = "Apply Byte-Pair Encoding (BPE) tokenization techniques to preprocess the input dataset for language modeling."
        co, sim, pos = self.mapper.map_question(q, bloom_level="Apply")
        self.assertEqual(co, "CO1")
        self.assertGreater(sim, 0.3)
        self.assertEqual(pos, ["PO1"])

    def test_standalone_map_question_to_co(self):
        """Verify the standalone map_question_to_co function interface."""
        q = "Evaluate different quantization schemes (AWQ vs GPTQ) for LLM model compression."
        co, sim, pos = map_question_to_co(q, bloom_level="Evaluate")
        self.assertEqual(co, "CO3")
        self.assertIsInstance(sim, float)
        self.assertIsInstance(pos, list)
        self.assertEqual(pos, ["PO3", "PO4"])

    def test_custom_cos_dynamic_adaptation(self):
        """Verify mapper works when custom COs from an uploaded syllabus are supplied."""
        custom_cos = {
            "CO1": "Understand the fundamentals of deep neural network architectures.",
            "CO2": "Implement convolutional neural networks for visual pattern recognition.",
            "CO3": "Evaluate generative adversarial networks for image synthesis."
        }
        custom_matrix = {
            "CO1": {"PO1": 3},
            "CO2": {"PO2": 3, "PO3": 3},
            "CO3": {"PO4": 3, "PO5": 3}
        }
        q = "Evaluate the discriminator and generator loss in conditional GANs."
        co, sim, pos = map_question_to_co(
            q,
            bloom_level="Evaluate",
            cos=custom_cos,
            co_po_matrix=custom_matrix
        )
        self.assertEqual(co, "CO3")
        self.assertEqual(pos, ["PO4", "PO5"])


if __name__ == "__main__":
    unittest.main()

