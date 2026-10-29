"""Unit and Integration Tests for RAG Retrievers (Dense Semantic vs Sparse TF-IDF).

Course: Natural Language Processing for LLM (24AM7PCLLM)
Institution: B.M.S. College of Engineering (BMSCE)
Department: Artificial Intelligence & Machine Learning
"""

import unittest
import os
import shutil
import tempfile
from typing import List, Dict, Any

from modules.retriever import KnowledgeBase, TFIDFRetriever, compare_retrievers


class TestKnowledgeBaseAndTFIDF(unittest.TestCase):
    """Test suite for Dense FAISS and Sparse TF-IDF retrieval pipelines."""

    @classmethod
    def setUpClass(cls):
        """Prepare sample academic syllabus/textbook chunks for retrieval testing."""
        cls.sample_chunks: List[Dict[str, Any]] = [
            {
                "chunk_id": "test_chunk_1",
                "text": "The Transmission Control Protocol (TCP) provides reliable, ordered, and error-checked delivery of a stream of octets between applications running on hosts communicating via an IP network. It handles flow control using sliding window mechanisms and implements congestion control algorithms like slow start and congestion avoidance.",
                "source": "Networking_Textbook.pdf",
                "page": 45
            },
            {
                "chunk_id": "test_chunk_2",
                "text": "User Datagram Protocol (UDP) is a connectionless transport layer protocol that provides a simple transmission model without implicit hand-shaking dialogues for providing reliability, ordering, or data integrity. It is widely used in real-time multimedia streaming and VoIP services.",
                "source": "Networking_Textbook.pdf",
                "page": 49
            },
            {
                "chunk_id": "test_chunk_3",
                "text": "Public-key cryptography, or asymmetric cryptography, is a cryptographic system that uses pairs of keys: public keys, which may be disseminated widely, and private keys, which are known only to the owner. RSA and Diffie-Hellman are foundational algorithms ensuring secure digital signatures and secret key exchange.",
                "source": "Security_Notes.pdf",
                "page": 12
            },
            {
                "chunk_id": "test_chunk_4",
                "text": "Convolutional Neural Networks (CNNs) are regularized versions of multilayer perceptrons that use convolution operations instead of general matrix multiplication in at least one of their layers. They feature pooling layers, feature maps, and are extensively utilized in computer vision and image classification tasks.",
                "source": "Deep_Learning_Syllabus.pdf",
                "page": 3
            },
            {
                "chunk_id": "test_chunk_5",
                "text": "Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks are designed to capture temporal dependencies in sequential data such as natural language text, speech signals, and time series predictions, overcoming the vanishing gradient problem through gating cells.",
                "source": "Deep_Learning_Syllabus.pdf",
                "page": 6
            }
        ]
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_dense_semantic_retrieval(self):
        """Verify that KnowledgeBase retrieves the correct chunk via dense semantic similarity."""
        kb = KnowledgeBase()
        count = kb.build_from_chunks(self.sample_chunks)
        self.assertEqual(count, len(self.sample_chunks))

        # Query about congestion avoidance and flow control
        results = kb.retrieve("How does TCP regulate congestion and ensure reliable byte streams?", k=2)
        self.assertGreater(len(results), 0)
        top_text, top_score, top_source, top_page = results[0]
        self.assertIn("Transmission Control Protocol", top_text)
        self.assertEqual(top_source, "Networking_Textbook.pdf")
        self.assertEqual(top_page, 45)
        # Cosine similarity for normalized vectors is bounded [-1, 1], highly relevant matches are > 0.4
        self.assertGreater(top_score, 0.4)

    def test_dense_semantic_matching_without_exact_keywords(self):
        """Demonstrate that dense embeddings capture synonyms (e.g., 'asymmetric encryption' -> 'public-key cryptography')."""
        kb = KnowledgeBase()
        kb.build_from_chunks(self.sample_chunks)

        # Note: 'asymmetric encryption' does not appear verbatim in chunk text, but 'asymmetric cryptography' does
        results = kb.retrieve("asymmetric encryption and digital signature algorithms", k=1)
        self.assertGreater(len(results), 0)
        top_text, _, _, _ = results[0]
        self.assertIn("Public-key cryptography", top_text)

    def test_tfidf_retrieval(self):
        """Verify that TFIDFRetriever retrieves relevant chunks based on term frequency."""
        tfidf = TFIDFRetriever()
        count = tfidf.build_from_chunks(self.sample_chunks)
        self.assertEqual(count, len(self.sample_chunks))

        results = tfidf.retrieve("Convolutional Neural Networks pooling layers", k=2)
        self.assertGreater(len(results), 0)
        top_text, top_score, _, _ = results[0]
        self.assertIn("Convolutional Neural Networks", top_text)
        self.assertGreater(top_score, 0.0)

    def test_persistence_save_load(self):
        """Verify that indices can be saved to disk and loaded with identical results."""
        kb = KnowledgeBase()
        kb.build_from_chunks(self.sample_chunks)
        save_path = os.path.join(self.temp_dir, "kb_store")
        kb.save(save_path)

        loaded_kb = KnowledgeBase()
        loaded_kb.load(save_path)
        self.assertEqual(len(loaded_kb.chunks), len(self.sample_chunks))

        results_orig = kb.retrieve("sequential data temporal dependencies", k=1)
        results_loaded = loaded_kb.retrieve("sequential data temporal dependencies", k=1)

        self.assertEqual(results_orig[0][0], results_loaded[0][0])
        self.assertAlmostEqual(results_orig[0][1], results_loaded[0][1], places=4)

    def test_retriever_comparison_metrics(self):
        """Verify compare_retrievers calculates Precision@k, Recall@k, and MRR correctly."""
        kb = KnowledgeBase()
        kb.build_from_chunks(self.sample_chunks)

        tfidf = TFIDFRetriever()
        tfidf.build_from_chunks(self.sample_chunks)

        test_queries = [
            "reliable byte stream TCP protocol",
            "computer vision convolutional feature maps"
        ]
        ground_truth = [
            ["tcp", "reliable", "stream"],
            ["convolutional", "cnn", "pooling"]
        ]

        report = compare_retrievers(test_queries, ground_truth, kb, tfidf, k=2)

        self.assertEqual(report["total_queries_evaluated"], 2)
        self.assertEqual(report["k"], 2)
        self.assertIn("dense_embeddings_all_MiniLM_L6_v2", report)
        self.assertIn("sparse_lexical_TFIDF", report)

        dense_stats = report["dense_embeddings_all_MiniLM_L6_v2"]
        self.assertGreater(dense_stats["mean_precision_at_k"], 0.0)
        self.assertGreater(dense_stats["mean_recall_at_k"], 0.0)
        self.assertGreater(dense_stats["mean_reciprocal_rank_MRR"], 0.0)

    def test_compare_retrievers_default_signature(self):
        """Verify compare_retrievers can be called directly with (queries, ground_truth)."""
        # Save current KB and TF-IDF to index store
        kb = KnowledgeBase()
        kb.build_from_chunks(self.sample_chunks)
        kb.save("data/index_store")

        tfidf = TFIDFRetriever()
        tfidf.build_from_chunks(self.sample_chunks)
        tfidf.save("data/index_store")

        queries = ["UDP connectionless audio video streaming"]
        ground_truth = [["udp", "connectionless", "multimedia"]]

        # Call with only (queries, ground_truth)
        report = compare_retrievers(queries, ground_truth)
        self.assertEqual(report["total_queries_evaluated"], 1)
        self.assertIn("dense_embeddings_all_MiniLM_L6_v2", report)


if __name__ == "__main__":
    unittest.main()

