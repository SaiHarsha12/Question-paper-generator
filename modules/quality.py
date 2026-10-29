"""Question Paper Quality and Grounding Evaluation Subsystem.

Course: Natural Language Processing for LLM (24AM7PCLLM)
Institution: B.M.S. College of Engineering (BMSCE)
Department: Artificial Intelligence & Machine Learning

This module implements comprehensive automated quality assurance for generated exam questions:
1. Relevance Score: Cosine similarity between question embedding and mean of retrieved chunk embeddings.
2. Duplicate Check: Pairwise cosine similarity matrix within the paper to flag near-duplicates (threshold >= 0.85).
3. Groundedness Score: Fraction of question content words (lemmatized, stopwords removed) present in retrieved context.
4. Format Check: Verifies starting action verbs, absence of introductory chatter, length constraints (8-80 words), and absence of marks text.
5. Unit Coverage: Confirms complete unit distribution and topic sampling.
6. Quality Report: Comprehensive aggregate audit for both individual questions and full examination papers.
"""

from typing import List, Dict, Tuple, Optional, Any, Union, Set
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from modules.preprocess import tokenize, remove_stopwords, lemmatize, stem
from modules.co_mapper import get_embedding_model
from nltk.stem import WordNetLemmatizer

# Comprehensive action verbs for BMSCE / VTU question papers across Bloom cognitive levels
VALID_ACTION_VERBS: Set[str] = {
    # Remember (L1)
    "define", "list", "state", "recall", "name", "mention", "identify", "label",
    # Understand (L2)
    "explain", "describe", "discuss", "summarize", "outline", "classify", "elaborate",
    "categorize", "interpret", "distinguish", "paraphrase",
    # Apply (L3)
    "solve", "illustrate", "demonstrate", "apply", "calculate", "determine", "compute",
    "show", "execute", "implement", "construct",
    # Analyze (L4)
    "compare", "differentiate", "contrast", "analyze", "examine", "investigate",
    "correlate", "categorize", "deconstruct",
    # Evaluate (L5)
    "justify", "critique", "assess", "evaluate", "appraise", "validate", "defend",
    "argue", "recommend", "judge",
    # Create (L6)
    "design", "formulate", "develop", "devise", "propose", "synthesize", "generate",
    "compose", "derive", "plan"
}

# Multi-word verb stems common in VTU engineering question papers
VALID_VERB_PREFIXES: List[str] = [
    "explain the", "illustrate the", "demonstrate the", "differentiate between",
    "compare and contrast", "list and elaborate", "discuss in detail", "explain how",
    "describe the", "evaluate the", "assess the", "analyze the", "design an",
    "design a", "formulate a", "derive an", "derive the", "calculate the"
]


def relevance_score(
    question: str,
    retrieved_chunks: List[Union[Dict[str, Any], Tuple[Any, ...], str]],
    model_name: str = "all-MiniLM-L6-v2"
) -> float:
    """Calculate cosine similarity between question embedding and mean of retrieved context chunks.

    Args:
        question: Generated question text.
        retrieved_chunks: Chunks retrieved by the RAG retriever (dicts, tuples, or strings).
        model_name: Sentence-Transformers model name.

    Returns:
        Float cosine similarity in [0.0, 1.0]. Returns 1.0 if no chunks (baseline mode).
    """
    if not question.strip():
        return 0.0

    if not retrieved_chunks:
        # In baseline mode (no retrieval), relevance is neutral/baseline default
        return 1.0

    # Extract text from heterogeneous chunk formats
    chunk_texts: List[str] = []
    for c in retrieved_chunks:
        if isinstance(c, dict):
            text = c.get("text", "")
        elif isinstance(c, tuple) and len(c) > 0:
            text = str(c[0])
        elif isinstance(c, str):
            text = c
        else:
            text = str(c)
        if text.strip():
            chunk_texts.append(text.strip())

    if not chunk_texts:
        return 1.0

    model = get_embedding_model(model_name)

    # Encode question and chunks with L2 normalization
    q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    c_embs = model.encode(chunk_texts, convert_to_numpy=True, normalize_embeddings=True)

    # Calculate centroid / mean of retrieved chunk embeddings
    mean_chunk_emb = np.mean(c_embs, axis=0)
    norm = np.linalg.norm(mean_chunk_emb)
    if norm > 0:
        mean_chunk_emb = mean_chunk_emb / norm

    similarity = float(np.dot(q_emb, mean_chunk_emb))
    return round(max(0.0, min(1.0, similarity)), 4)


def duplicate_check(
    questions: List[str],
    threshold: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2"
) -> List[Dict[str, Any]]:
    """Perform pairwise cosine similarity audit across questions to identify near-duplicates.

    Args:
        questions: List of question texts in the examination paper.
        threshold: Cosine similarity cutoff for flagging duplicates (default 0.85).
        model_name: Sentence-Transformers model name.

    Returns:
        List of flagged duplicate pair records: [{idx1, idx2, q1, q2, similarity}].
    """
    if len(questions) < 2:
        return []

    model = get_embedding_model(model_name)
    embs = model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
    sim_matrix = np.dot(embs, embs.T)

    flagged: List[Dict[str, Any]] = []
    n = len(questions)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= threshold:
                flagged.append({
                    "idx1": i,
                    "idx2": j,
                    "q1": questions[i],
                    "q2": questions[j],
                    "similarity": round(sim, 4)
                })

    return flagged


def is_candidate_duplicate(
    candidate: str,
    existing_questions: List[str],
    threshold: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2"
) -> Tuple[bool, float, int]:
    """Check if a newly generated candidate question is a duplicate of any existing question.

    Args:
        candidate: Candidate question text.
        existing_questions: Already accepted questions in the paper.
        threshold: Duplicate similarity threshold.
        model_name: Sentence-Transformers model name.

    Returns:
        Tuple of (is_dup: bool, max_sim: float, matched_idx: int).
    """
    if not existing_questions or not candidate.strip():
        return False, 0.0, -1

    model = get_embedding_model(model_name)
    c_emb = model.encode([candidate], convert_to_numpy=True, normalize_embeddings=True)[0]
    ex_embs = model.encode(existing_questions, convert_to_numpy=True, normalize_embeddings=True)

    sims = np.dot(ex_embs, c_emb)
    max_idx = int(np.argmax(sims))
    max_sim = float(sims[max_idx])

    return (max_sim >= threshold), round(max_sim, 4), max_idx


def groundedness_score(
    question: str,
    retrieved_chunks: Union[List[Union[Dict[str, Any], Tuple[Any, ...], str]], str]
) -> float:
    """Calculate the fraction of question content words grounded in the retrieved context.

    Uses NLTK tokenization, stopword removal, and WordNet lemmatization from modules.preprocess.

    Args:
        question: Question text.
        retrieved_chunks: Retrieved context chunks (list of dicts/tuples/strings, or unified string).

    Returns:
        Float groundedness fraction in [0.0, 1.0]. Returns 1.0 if no context given (baseline mode).
    """
    if not question.strip():
        return 0.0

    # Build context string
    if isinstance(retrieved_chunks, str):
        context_str = retrieved_chunks
    elif isinstance(retrieved_chunks, list):
        parts: List[str] = []
        for c in retrieved_chunks:
            if isinstance(c, dict):
                parts.append(c.get("text", ""))
            elif isinstance(c, tuple) and len(c) > 0:
                parts.append(str(c[0]))
            else:
                parts.append(str(c))
        context_str = " ".join(parts).strip()
    else:
        context_str = str(retrieved_chunks).strip()

    if not context_str:
        # In baseline mode (no retrieved context), return 1.0 (not penalized for missing context)
        return 1.0

    # Extract question content words (tokenized, stopwords removed, lemmatized)
    q_tokens = tokenize(question)
    q_filtered = remove_stopwords(q_tokens)
    q_lemmas = lemmatize(q_filtered)

    # Filter out pure punctuation or short symbols
    content_words = [
        w.lower() for w in q_lemmas
        if len(w) > 1 and not re.match(r"^[^a-zA-Z0-9]+$", w)
    ]

    if not content_words:
        return 1.0

    # Preprocess context into lemma lookup set, verb lemma set, stems, and lowercased string
    wnl = WordNetLemmatizer()
    ctx_tokens = tokenize(context_str)
    ctx_filtered = remove_stopwords(ctx_tokens)
    ctx_lemmas = set(w.lower() for w in lemmatize(ctx_filtered))
    for w in ctx_filtered:
        try:
            ctx_lemmas.add(wnl.lemmatize(w.lower(), pos="v"))
        except Exception:
            pass
    ctx_stems = set(stem(ctx_filtered))
    ctx_lower = context_str.lower()

    # Grounded if lemma is in context lemmas, verb lemma matches, stem matches, or substring in context
    grounded_count = 0
    for w in content_words:
        w_lower = w.lower()
        w_stem = stem([w_lower])[0]
        try:
            w_verb = wnl.lemmatize(w_lower, pos="v")
        except Exception:
            w_verb = w_lower

        matched = (
            w_lower in ctx_lemmas or
            w_verb in ctx_lemmas or
            w_stem in ctx_stems or
            w_lower in ctx_lower or
            w_verb in ctx_lower
        )

        # Handle hyphenated technical terms (e.g. self-attention, parameter-efficient)
        if not matched and "-" in w_lower:
            parts = [p for p in w_lower.split("-") if len(p) > 2]
            if parts and any(p in ctx_lemmas or p in ctx_lower for p in parts):
                matched = True

        if matched:
            grounded_count += 1

    return round(float(grounded_count / len(content_words)), 4)


def format_check(question: str) -> Dict[str, Any]:
    """Verify that a question meets university exam format standards.

    Checks:
    1. Action verb prefix: starts with a valid Bloom or exam action verb.
    2. No conversational intro: rejects 'Here is', 'Sure,', 'Question:', etc.
    3. Length bounds: between 8 and 80 words.
    4. No marks clutter: rejects embedded mark annotations like '(10 marks)'.

    Args:
        question: Question text to evaluate.

    Returns:
        Dict with 'valid' (bool), 'issues' (list of str), and sub-check booleans.
    """
    clean_q = question.strip()
    issues: List[str] = []

    # 1. No conversational intro
    intro_pattern = r"^(?:here\s+is|here\s+are|sure|certainly|below\s+is|this\s+question|as\s+requested|question\s*\d*[:.-]?)\b"
    has_intro = bool(re.search(intro_pattern, clean_q, re.I))
    if has_intro:
        issues.append("Contains conversational preamble (e.g. 'Here is', 'Question:')")

    # 2. Word count between 8 and 80 words
    words = clean_q.split()
    word_count = len(words)
    length_valid = (8 <= word_count <= 80)
    if word_count < 8:
        issues.append(f"Question too brief ({word_count} words; minimum is 8)")
    elif word_count > 80:
        issues.append(f"Question exceeds maximum length ({word_count} words; maximum is 80)")

    # 3. Action verb check
    starts_with_verb = False
    clean_lower = clean_q.lower()

    # Check multi-word prefixes first
    for prefix in VALID_VERB_PREFIXES:
        if clean_lower.startswith(prefix):
            starts_with_verb = True
            break

    # Check first word against valid action verbs
    if not starts_with_verb and words:
        first_word = re.sub(r"[^a-zA-Z]", "", words[0]).lower()
        if first_word in VALID_ACTION_VERBS:
            starts_with_verb = True

    if not starts_with_verb:
        issues.append(f"Does not start with a recognized Bloom action verb (starts with '{words[0] if words else ''}')")

    # 4. No marks text
    marks_pattern = r"(?:\(\s*\d+\s*marks?\s*\)|\[\s*\d+\s*marks?\s*\]|\b\d+\s*marks?\b)"
    has_marks = bool(re.search(marks_pattern, clean_q, re.I))
    if has_marks:
        issues.append("Contains embedded marks text (marks must only appear in the table Marks column)")

    valid = (not has_intro) and length_valid and starts_with_verb and (not has_marks)

    return {
        "valid": valid,
        "issues": issues,
        "starts_with_verb": starts_with_verb,
        "no_intro": not has_intro,
        "length_valid": length_valid,
        "word_count": word_count,
        "no_marks": not has_marks
    }


def unit_coverage(
    paper: List[Dict[str, Any]],
    expected_units: Optional[List[str]] = None,
    expected_per_unit: int = 4
) -> Dict[str, Any]:
    """Verify that every unit has the required question count and all topics were sampled.

    Args:
        paper: List of question logs or dicts with 'unit' and 'topic' fields.
        expected_units: Optional list of expected unit identifiers (e.g. ['UNIT-1', 'UNIT-2']).
        expected_per_unit: Number of questions expected per unit (default 4 for VTU choice model).

    Returns:
        Coverage report dict with completeness flag, breakdown, and issues.
    """
    breakdown: Dict[str, Dict[str, Any]] = {}
    for item in paper:
        u = item.get("unit", "Unknown")
        topic = item.get("topic", "N/A")
        if u not in breakdown:
            breakdown[u] = {
                "count": 0,
                "expected": expected_per_unit,
                "subtopics_attempted": [],
                "complete": False
            }
        breakdown[u]["count"] += 1
        if topic not in breakdown[u]["subtopics_attempted"]:
            breakdown[u]["subtopics_attempted"].append(topic)

    # Check completeness
    issues: List[str] = []
    units_to_check = expected_units if expected_units else list(breakdown.keys())
    complete_units = 0

    for u in units_to_check:
        if u not in breakdown:
            issues.append(f"Unit {u} has no generated questions")
            continue
        c = breakdown[u]["count"]
        if c == expected_per_unit:
            breakdown[u]["complete"] = True
            complete_units += 1
        elif c < expected_per_unit:
            issues.append(f"Unit {u} has {c}/{expected_per_unit} questions (incomplete)")
        else:
            breakdown[u]["complete"] = True
            complete_units += 1

    total_expected = len(units_to_check) * expected_per_unit
    total_generated = sum(b["count"] for b in breakdown.values())

    return {
        "complete": len(issues) == 0 and total_generated >= total_expected,
        "total_questions": total_generated,
        "expected_questions": total_expected,
        "units_covered": complete_units,
        "total_units": len(units_to_check),
        "unit_breakdown": breakdown,
        "issues": issues
    }


def quality_report(
    paper: List[Dict[str, Any]],
    threshold_duplicate: float = 0.85
) -> Dict[str, Any]:
    """Generate a complete quality audit report across all questions in an exam paper.

    Args:
        paper: List of question records containing 'question', 'retrieved_chunks', 'unit', 'topic'.
        threshold_duplicate: Pairwise cosine similarity cutoff.

    Returns:
        Dictionary containing per-question metrics table and aggregate paper-level summary statistics.
    """
    if not paper:
        return {
            "summary": {
                "total_questions": 0,
                "avg_relevance": 0.0,
                "avg_groundedness": 0.0,
                "format_compliance_rate": 0.0,
                "duplicate_count": 0,
                "overall_passed": False
            },
            "per_question": [],
            "duplicate_pairs": [],
            "coverage": {}
        }

    question_texts = [p.get("question", "") for p in paper]

    # 1. Pairwise Duplicate Analysis
    dup_pairs = duplicate_check(question_texts, threshold=threshold_duplicate)
    flagged_indices: Set[int] = set()
    for dp in dup_pairs:
        flagged_indices.add(dp["idx1"])
        flagged_indices.add(dp["idx2"])

    # 2. Per-question evaluations
    per_question_rows: List[Dict[str, Any]] = []
    total_rel = 0.0
    total_grounded = 0.0
    format_passes = 0

    for idx, item in enumerate(paper):
        q = item.get("question", "")
        chunks = item.get("retrieved_chunks", [])
        qno = item.get("qno", idx + 1)
        unit = item.get("unit", "N/A")
        topic = item.get("topic", "N/A")

        # Compute individual metrics
        rel = relevance_score(q, chunks)
        grounded = groundedness_score(q, chunks)
        fmt = format_check(q)
        is_dup = (idx in flagged_indices)

        total_rel += rel
        total_grounded += grounded
        if fmt["valid"]:
            format_passes += 1

        per_question_rows.append({
            "qno": qno,
            "unit": unit,
            "topic": topic,
            "relevance": rel,
            "groundedness": grounded,
            "format_valid": "✅ Pass" if fmt["valid"] else "❌ Fail",
            "format_issues": "; ".join(fmt["issues"]) if fmt["issues"] else "None",
            "duplicate_risk": "⚠️ High Risk" if is_dup else "✅ Unique",
            "word_count": fmt["word_count"],
            "question": q
        })

    n = len(paper)
    avg_rel = round(total_rel / max(n, 1), 4)
    avg_grounded = round(total_grounded / max(n, 1), 4)
    fmt_rate = round((format_passes / max(n, 1)) * 100, 1)

    # 3. Unit coverage
    coverage = unit_coverage(paper)

    overall_passed = (
        len(dup_pairs) == 0 and
        fmt_rate >= 85.0 and
        avg_grounded >= 0.6 and
        coverage.get("complete", False)
    )

    return {
        "summary": {
            "total_questions": n,
            "avg_relevance": avg_rel,
            "avg_groundedness": avg_grounded,
            "format_compliance_rate": f"{fmt_rate}%",
            "duplicate_count": len(dup_pairs),
            "coverage_complete": coverage.get("complete", False),
            "overall_passed": overall_passed
        },
        "per_question": per_question_rows,
        "duplicate_pairs": dup_pairs,
        "coverage": coverage
    }
