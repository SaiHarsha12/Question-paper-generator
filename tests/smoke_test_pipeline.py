"""End-to-End Smoke Test for 24AM7PCLLM Question Paper Generator Pipeline.

This script executes a complete walkthrough of every core pipeline component:
1. Preprocessor & Text Normalization
2. Table & Unit Syllabus Extractor
3. FAISS & TF-IDF Knowledge Retrieval
4. RAG Question Generation (Ollama / Fallback)
5. Bloom Cognitive Classification
6. Course Outcome (CO) & PO Accreditation Mapping
7. Automated Quality & Grounding Audit
8. Exam Paper PDF Generation (Modern fpdf2 API)
9. Evaluation Test Cases & Observations Registry Loading
"""

import os
import sys

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.preprocess import (
    clean_text,
    tokenize,
    chunk_text,
    preprocessing_report
)
from modules.extractor import get_unit_concepts
from modules.retriever import KnowledgeBase, TFIDFRetriever, compare_retrievers
from modules.rag_engine import generate_exam_question
from modules.bloom_classifier import get_bloom_classifier
from modules.co_mapper import map_question_to_co, load_course_config
from modules.quality import (
    format_check,
    is_candidate_duplicate,
    relevance_score,
    groundedness_score,
    quality_report
)
from modules.pdf_builder import ExamPaperPDF
from evaluation.run_evaluation import load_test_cases
from scripts.log_observation import load_observations


def run_smoke_test():
    print("=" * 70)
    print("[SMOKE TEST] Starting End-to-End Pipeline Smoke Test (24AM7PCLLM)")
    print("=" * 70)
    failures = []

    # 1. Preprocessor Smoke Test
    print("\n[1/9] Testing Text Preprocessor & Normalization...")
    try:
        raw_sample = "BMSCE Syllabus 2026\nUnit - 1: Subword token-\nization algorithms.\nPage 1 of 12"
        cleaned = clean_text(raw_sample)
        assert "tokenization" in cleaned, "Hyphenated line break not fixed"
        assert "Page" not in cleaned, "Page number not removed"
        tokens = tokenize(cleaned)
        assert len(tokens) > 0, "Tokenization returned empty list"
        chunks = chunk_text(cleaned, chunk_size=200, overlap=50)
        assert len(chunks) > 0, "Chunking returned empty list"
        rep = preprocessing_report(raw_sample)
        assert "raw_character_count" in rep, "Preprocessing report missing keys"
        print("  [PASS] Preprocessor passed cleanly.")
    except Exception as e:
        print(f"  [FAIL] Preprocessor failed: {e}")
        failures.append(f"Preprocessor: {e}")

    # 2. Extractor Smoke Test (Table & Unit Parsing)
    print("\n[2/9] Testing Syllabus Extractor...")
    try:
        sample_syllabus = """
        UNIT - 1
        Byte-Pair Encoding, WordPiece Algorithm, Subword tokenization
        10 Hrs.
        UNIT - 2
        Scaled Dot-Product Attention, Transformers, Multi-Head Attention
        10 Hrs.
        CO1: Apply data processing techniques on NLP inputs.
        CO2: Analyze computational linguistic patterns.
        """
        units = get_unit_concepts(sample_syllabus)
        assert "UNIT-1" in units, f"Units not parsed: {units.keys()}"
        full_data = get_unit_concepts(sample_syllabus, return_full=True)
        assert "cos" in full_data and len(full_data["cos"]) >= 2, "COs not parsed"
        
        # Test table-style course plan parser directly
        from modules.extractor import _extract_table_course_plan
        table_mock = [
            [
                ["Week", "Class", "Unit", "Topic"],
                ["Week 1", "Class 1", "1", "Introduction to Recurrent Neural Networks"],
                ["Week 2", "Class 2", "2", "Transformer Self-Attention"]
            ]
        ]
        t_units = _extract_table_course_plan(table_mock)
        assert "UNIT-1" in t_units and "UNIT-2" in t_units, "Table course plan not parsed"
        print("  [PASS] Extractor passed cleanly.")
    except Exception as e:
        print(f"  [FAIL] Extractor failed: {e}")
        failures.append(f"Extractor: {e}")

    # 3. Retriever Smoke Test (Dense FAISS & TF-IDF)
    print("\n[3/9] Testing Dense FAISS & TF-IDF Knowledge Retrieval...")
    try:
        kb = KnowledgeBase()
        kb.chunks = [
            {"chunk_id": "c1", "text": "Byte-Pair Encoding (BPE) iteratively merges frequent pairs of bytes or characters.", "source": "unit1.pdf", "page": 1},
            {"chunk_id": "c2", "text": "Scaled dot-product attention divides dot products by the square root of the key dimension.", "source": "unit2.pdf", "page": 2},
            {"chunk_id": "c3", "text": "Low-Rank Adaptation (LoRA) freezes the pre-trained model weights and injects trainable rank decomposition matrices.", "source": "unit3.pdf", "page": 3}
        ]
        kb.embeddings = kb.model.encode([c["text"] for c in kb.chunks], normalize_embeddings=True)
        import faiss
        import numpy as np
        kb.index = faiss.IndexFlatIP(kb.embeddings.shape[1])
        kb.index.add(np.ascontiguousarray(kb.embeddings.astype(np.float32)))

        results = kb.retrieve("How does LoRA adapt model parameters?", k=1)
        assert len(results) == 1, "Dense retrieval returned no results"
        assert "Low-Rank Adaptation" in results[0][0], f"Incorrect chunk retrieved: {results[0][0]}"

        tfidf = TFIDFRetriever()
        tfidf.fit(kb.chunks)
        tfidf_results = tfidf.retrieve("Byte-Pair Encoding", k=1)
        assert len(tfidf_results) == 1, "TF-IDF retrieval returned no results"

        eval_ir = compare_retrievers(
            queries=["Byte-Pair Encoding"],
            ground_truth=[["c1"]],
            dense_retriever=kb,
            tfidf_retriever=tfidf,
            k=2
        )
        assert "dense" in eval_ir and "tfidf" in eval_ir, "compare_retrievers missing output keys"
        print("  [PASS] Retriever passed cleanly.")
    except Exception as e:
        print(f"  [FAIL] Retriever failed: {e}")
        failures.append(f"Retriever: {e}")

    # 4. RAG Engine Smoke Test
    print("\n[4/9] Testing RAG Engine (Few-Shot & CoT Check)...")
    try:
        out_rag = generate_exam_question(
            subtopic="Byte-Pair Encoding and WordPiece subword tokenization",
            bloom_level="understand",
            question_type="theory",
            difficulty="Moderate",
            kb=kb,
            mode="rag",
            prompt_style="few_shot",
            model_name="llama3",
            k=2
        )
        assert "question" in out_rag and len(out_rag["question"]) > 10, "RAG generation returned empty question"
        assert len(out_rag["retrieved_chunks"]) > 0, "RAG mode failed to retrieve chunks"
        print(f"  Sample Generated Question: {out_rag['question'][:75]}...")
        print("  [PASS] RAG Engine passed cleanly.")
    except Exception as e:
        print(f"  [FAIL] RAG Engine failed: {e}")
        failures.append(f"RAG Engine: {e}")

    # 5. Bloom Classifier Smoke Test
    print("\n[5/9] Testing Bloom Cognitive Classifier...")
    try:
        classifier = get_bloom_classifier("best")
        pred_level = classifier.predict("Differentiate between Byte-Pair Encoding and WordPiece with neat examples.")
        assert pred_level in ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"], f"Invalid Bloom level: {pred_level}"
        print(f"  Predicted Level for 'Differentiate...': {pred_level}")
        print("  [PASS] Bloom Classifier passed cleanly.")
    except Exception as e:
        print(f"  [FAIL] Bloom Classifier failed: {e}")
        failures.append(f"Bloom Classifier: {e}")

    # 6. Course Outcome & PO Accreditation Mapper Smoke Test
    print("\n[6/9] Testing CO & PO Accreditation Mapper...")
    try:
        co, sim, pos = map_question_to_co(
            question="Analyze the multi-head self-attention mechanism in the Transformer encoder architecture.",
            bloom_level="Analyze",
            blend_prior=True
        )
        assert co in ["CO1", "CO2", "CO3", "CO4"], f"Invalid mapped CO: {co}"
        assert len(pos) > 0, "No POs mapped"
        print(f"  Mapped Outcome: {co} (Similarity: {sim:.3f}) | Mapped POs: {', '.join(pos)}")
        print("  [PASS] CO/PO Mapper passed cleanly.")
    except Exception as e:
        print(f"  [FAIL] CO/PO Mapper failed: {e}")
        failures.append(f"CO/PO Mapper: {e}")

    # 7. Quality & Grounding Audit Smoke Test
    print("\n[7/9] Testing Quality & Grounding Audit Subsystem...")
    try:
        q_test = "Explain the fundamental mechanism of Byte-Pair Encoding with a neat diagram."
        fmt = format_check(q_test)
        assert fmt["valid"], f"Compliant question failed format check: {fmt['issues']}"
        chunks_text = [c["text"] for c in kb.chunks[:1]]
        g_score = groundedness_score(q_test, chunks_text)
        r_score = relevance_score(q_test, chunks_text)
        assert 0.0 <= g_score <= 1.0, "Groundedness score out of bounds"
        assert -1.0 <= r_score <= 1.0, "Relevance score out of bounds"
        is_dup, _, _ = is_candidate_duplicate(q_test, [q_test], threshold=0.85)
        assert is_dup, "Exact duplicate not flagged"
        print("  [PASS] Quality & Grounding Audit passed cleanly.")
    except Exception as e:
        print(f"  [FAIL] Quality Audit failed: {e}")
        failures.append(f"Quality Audit: {e}")

    # 8. PDF Builder Smoke Test (Modern fpdf2 API)
    print("\n[8/9] Testing PDF Builder (fpdf2 new_x/new_y API)...")
    try:
        pdf = ExamPaperPDF()
        pdf.add_page()
        pdf.draw_first_page_header(
            branch="Artificial Intelligence and Machine Learning",
            code="24AM7PCLLM",
            course="Natural Language Processing for LLM",
            sem="VII"
        )
        pdf.unit_header("UNIT - 1")
        pdf.question_row("1", "Explain the iterative vocabulary merging process in Byte-Pair Encoding.", "CO1", "PO1, PSO3", "10")
        pdf.question_row("2", "Discuss WordPiece tokenization and its scoring function with an example.", "CO1", "PO1", "10")
        pdf.or_row()
        pdf.question_row("3", "Compare Byte-Pair Encoding, WordPiece, and SentencePiece tokenization schemes.", "CO2", "PO2, PSO3", "10")
        pdf.question_row("4", "Illustrate how out-of-vocabulary tokens are handled in subword algorithms.", "CO1", "PO1", "10")
        
        os.makedirs("samples", exist_ok=True)
        sample_out = os.path.join("samples", "smoke_test_output.pdf")
        pdf.output(sample_out)
        assert os.path.exists(sample_out) and os.path.getsize(sample_out) > 1000, "PDF not created or empty"
        print(f"  [PASS] PDF Builder passed cleanly. Generated: {sample_out}")
    except Exception as e:
        print(f"  [FAIL] PDF Builder failed: {e}")
        failures.append(f"PDF Builder: {e}")

    # 9. Academic Test Cases & Observations Registry Smoke Test
    print("\n[9/9] Testing Test Cases & Observations Registry...")
    try:
        tcs = load_test_cases()
        assert len(tcs) >= 5, f"Expected at least 5 test cases, found {len(tcs)}"
        obs = load_observations()
        assert len(obs) >= 5, f"Expected at least 5 observations, found {len(obs)}"
        print(f"  Loaded {len(tcs)} defined academic test cases and {len(obs)} recorded observations.")
        print("  [PASS] Test Cases & Observations passed cleanly.")
    except Exception as e:
        print(f"  [FAIL] Test Cases & Observations failed: {e}")
        failures.append(f"Test Cases & Observations: {e}")

    # Summary
    print("\n" + "=" * 70)
    if not failures:
        print("[SUCCESS] ALL SMOKE TESTS PASSED CLEANLY! Whole pipeline is operational.")
    else:
        print(f"[WARNING] {len(failures)} SMOKE TEST(S) FAILED:")
        for f in failures:
            print(f"  - {f}")
    print("=" * 70)

    return len(failures) == 0


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
