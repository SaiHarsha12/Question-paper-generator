"""Unit and Integration Tests for NLP Preprocessing and Syllabus Extractor.

Course: Natural Language Processing for LLM (24AM7PCLLM)
Institution: B.M.S. College of Engineering (BMSCE)
Department: Artificial Intelligence & Machine Learning
"""

import unittest
from typing import List, Dict, Any

from modules.preprocess import (
    clean_text,
    tokenize,
    remove_stopwords,
    lemmatize,
    stem,
    chunk_text,
    preprocessing_report,
)
from modules.extractor import (
    _extract_heading_units,
    _extract_table_course_plan,
    _extract_course_outcomes,
    get_unit_concepts,
)


class TestCleanText(unittest.TestCase):
    """Test suite for text cleaning and Unicode normalization."""

    def test_unicode_normalization(self):
        raw = "“Artificial Intelligence”—it’s • revolutionary… \u00a0"
        cleaned = clean_text(raw)
        self.assertNotIn("“", cleaned)
        self.assertNotIn("”", cleaned)
        self.assertNotIn("—", cleaned)
        self.assertIn('"', cleaned)
        self.assertIn("-", cleaned)

    def test_hyphenated_line_breaks(self):
        raw = "Deep learn-\ning models are revolu-\ntionary."
        cleaned = clean_text(raw)
        self.assertIn("learning", cleaned)
        self.assertIn("revolutionary", cleaned)
        self.assertNotIn("learn-", cleaned)

    def test_page_number_removal(self):
        raw = "Heading\nPage 1 of 12\nSome syllabus text.\n4\nMore syllabus content."
        cleaned = clean_text(raw)
        self.assertNotIn("Page 1 of 12", cleaned)
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        self.assertNotIn("4", lines)

    def test_whitespace_normalization(self):
        raw = "Too   many    spaces   and\n\n\n\n\nextra blank lines."
        cleaned = clean_text(raw)
        self.assertNotIn("   ", cleaned)
        self.assertNotIn("\n\n\n", cleaned)


class TestTokenizationAndFiltering(unittest.TestCase):
    """Test suite for word tokenization and stopword removal."""

    def test_tokenize(self):
        sample = "Natural Language Processing with Deep Learning!"
        tokens = tokenize(sample, lower=True)
        self.assertEqual(tokens[0], "natural")
        self.assertEqual(tokens[1], "language")
        self.assertIn("deep", tokens)

    def test_remove_stopwords(self):
        tokens = ["the", "supervised", "and", "unsupervised", "learning", "in", "unit", "10", "hrs"]
        filtered = remove_stopwords(tokens)
        self.assertNotIn("the", filtered)
        self.assertNotIn("and", filtered)
        self.assertNotIn("in", filtered)
        self.assertNotIn("unit", filtered)
        self.assertNotIn("hrs", filtered)
        self.assertIn("supervised", filtered)
        self.assertIn("unsupervised", filtered)
        self.assertIn("learning", filtered)


class TestStemmingVsLemmatization(unittest.TestCase):
    """Academic comparison tests between Stemming and Lemmatization."""

    def test_stemming_vs_lemmatization(self):
        tokens = ["running", "studies", "computational", "networks"]
        lemmatized = lemmatize(tokens)
        stemmed = stem(tokens)

        # WordNet lemmatizer preserves valid dictionary roots
        self.assertEqual(lemmatized[0], "run")  # verb lemmatization
        self.assertEqual(lemmatized[1], "study")  # noun/verb lemmatization
        self.assertEqual(lemmatized[3], "network")

        # Porter stemmer uses heuristic suffix stripping
        self.assertEqual(stemmed[0], "run")
        self.assertEqual(stemmed[1], "studi")  # note the non-word stem 'studi'
        self.assertEqual(stemmed[2], "comput")  # non-word stem 'comput'

        # Demonstrates morphological differences
        self.assertNotEqual(lemmatized[1], stemmed[1])


class TestSentenceAwareChunking(unittest.TestCase):
    """Test suite for sentence-aware chunking with sliding window overlap."""

    def test_chunk_boundaries(self):
        doc = (
            "Sentence one is foundational. Sentence two describes architecture. "
            "Sentence three explains training dynamics. Sentence four evaluates performance."
        )
        chunks = chunk_text(doc, chunk_size=90, overlap=35, source="test_doc", page=1)
        self.assertGreaterEqual(len(chunks), 2)
        for c in chunks:
            self.assertTrue(c["chunk_id"].startswith("test_doc_chunk_"))
            self.assertEqual(c["source"], "test_doc")
            self.assertEqual(c["page"], 1)
            # Ensure chunks end cleanly with a period or complete sentence
            self.assertTrue(c["text"].endswith("."))

    def test_empty_chunking(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   "), [])


class TestPreprocessingReport(unittest.TestCase):
    """Test suite for preprocessing analytics report."""

    def test_report_metrics(self):
        text = (
            "Machine learning studies algorithmic methods. "
            "Deep neural networks have revolutionized natural language understanding."
        )
        report = preprocessing_report(text)
        required_keys = [
            "raw_character_count",
            "cleaned_character_count",
            "compression_ratio",
            "tokens_before_cleaning",
            "tokens_after_cleaning",
            "tokens_after_stopwords",
            "stopwords_removed",
            "vocabulary_size_raw",
            "vocabulary_size_lemmatized",
            "vocabulary_size_stemmed",
            "stemming_vs_lemmatization_samples",
            "total_chunks_generated"
        ]
        for key in required_keys:
            self.assertIn(key, report)
        self.assertGreater(report["raw_character_count"], 0)
        self.assertGreater(report["vocabulary_size_lemmatized"], 0)


class TestExtractorUpgrades(unittest.TestCase):
    """Test suite for syllabus heading, table-style, and CO extraction."""

    def test_heading_units(self):
        sample_syllabus = """
        UNIT - 1
        Foundations of Neural Networks, Perceptron, Multilayer Perceptron, Backpropagation
        10 Hrs.
        UNIT - 2
        Convolutional Neural Networks, Pooling, Padding, ResNet Architectures
        10 Hrs.
        """
        units = _extract_heading_units(sample_syllabus)
        self.assertIn("UNIT-1", units)
        self.assertIn("UNIT-2", units)
        self.assertTrue(any("Perceptron" in t for t in units["UNIT-1"]))
        self.assertTrue(any("Convolutional" in t for t in units["UNIT-2"]))

    def test_table_course_plan(self):
        # Mock table with Week | Class | Unit | Topic
        mock_tables = [
            [
                ["Week", "Class", "Unit", "Topic"],
                ["Week 1", "Class 1", "1", "Introduction to Recurrent Neural Networks"],
                ["Week 1", "Class 2", "1", "Vanishing and Exploding Gradient Problem"],
                ["Week 2", "Class 3", "2", "Attention Mechanisms and Transformers"],
                ["Week 2", "Class 4", "2", "Self-Attention and Multi-Head Attention"],
            ]
        ]
        units = _extract_table_course_plan(mock_tables)
        self.assertIn("UNIT-1", units)
        self.assertIn("UNIT-2", units)
        self.assertEqual(len(units["UNIT-1"]), 2)
        self.assertIn("Introduction to Recurrent Neural Networks", units["UNIT-1"])
        self.assertIn("Attention Mechanisms and Transformers", units["UNIT-2"])

    def test_course_outcomes_extraction(self):
        mock_tables = [
            [
                ["Course Outcomes", ""],
                ["CO1", "Understand foundational concepts of deep generative models."],
                ["CO2", "Design and train transformer-based language representations."],
            ]
        ]
        cos = _extract_course_outcomes("", mock_tables)
        self.assertIn("CO1", cos)
        self.assertIn("CO2", cos)
        self.assertTrue("deep generative" in cos["CO1"])

    def test_backward_compatibility(self):
        # Uses existing real sample PDF
        units = get_unit_concepts("data/CNS.pdf", return_full=False)
        self.assertIsInstance(units, dict)
        self.assertIn("UNIT-1", units)
        self.assertIsInstance(units["UNIT-1"], list)

        full = get_unit_concepts("data/CNS.pdf", return_full=True)
        self.assertIn("units", full)
        self.assertIn("cos", full)


if __name__ == "__main__":
    unittest.main()

