"""RAG Question Generation Engine for Academic Curriculum Synthesis.

Course: Natural Language Processing for LLM (24AM7PCLLM)
Institution: B.M.S. College of Engineering (BMSCE)
Department: Artificial Intelligence & Machine Learning

This module implements the generative synthesis half of the RAG pipeline:
1. Dual Generation Modes:
   - 'baseline': Direct zero/few-shot generation without context retrieval.
   - 'rag': Context-augmented generation with FAISS semantic retrieval injection.
2. Prompt Engineering Strategies:
   - 'zero_shot': Direct instructional prompt.
   - 'few_shot': In-context learning using 3 Bloom-calibrated university exam exemplars.
   - 'cot_check': Two-stage generation + self-verification audit for Bloom taxonomy alignment.
3. Bloom's Revised Taxonomy Action Verb Stems (Remember, Understand, Apply, Analyze, Evaluate, Create).
4. Configurable local model integration (llama3, mistral, phi3) with automatic retries and latency tracking.
"""

from typing import Dict, Any, List, Optional
import os
import re
import time
import json
import random

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    OllamaLLM = None

# Bloom's Revised Taxonomy action verb stems
BLOOM_VERBS: Dict[str, List[str]] = {
    "remember": ["Define", "List", "State", "Identify", "Recall"],
    "understand": ["Explain", "Describe", "Discuss in detail", "Elaborate on", "Illustrate"],
    "apply": ["Solve", "Demonstrate", "Illustrate the mechanism of", "Calculate", "Apply"],
    "analyze": ["Compare", "Differentiate between", "Analyze the structural differences between", "Distinguish"],
    "evaluate": ["Justify", "Critique", "Assess the limitations of", "Evaluate the efficiency of"],
    "create": ["Design", "Formulate", "Construct", "Devise an end-to-end"]
}

# Cache for Ollama LLM client instances
_LLM_CACHE: Dict[str, Any] = {}

# Path to few-shot exemplars
FEW_SHOT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "few_shot_examples.json"
)


def _get_llm(model_name: str = "llama3") -> Optional[Any]:
    """Retrieve or instantiate a cached OllamaLLM instance."""
    if OllamaLLM is None:
        return None
    if model_name not in _LLM_CACHE:
        try:
            _LLM_CACHE[model_name] = OllamaLLM(model=model_name, streaming=False)
        except Exception as e:
            print(f"[rag_engine] Warning: Could not instantiate OllamaLLM({model_name}): {e}")
            return None
    return _LLM_CACHE[model_name]


def _load_few_shot_examples(bloom_level: str) -> List[Dict[str, str]]:
    """Load 3 curated few-shot exam questions for a given Bloom level."""
    level = bloom_level.lower()
    if os.path.exists(FEW_SHOT_PATH):
        try:
            with open(FEW_SHOT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if level in data and isinstance(data[level], list):
                    return data[level][:3]
        except Exception as e:
            print(f"[rag_engine] Notice: Failed to load few-shot examples from file: {e}")

    # Fallback built-in exemplars
    return [
        {
            "subtopic": f"Core concepts of {level}",
            "question": f"Explain the fundamental mechanism and working principles of {level} in modern systems with a neat diagram."
        }
    ]


def _select_action_verb(bloom_level: str) -> str:
    """Select a pedagogically appropriate action verb for the Bloom level."""
    level = bloom_level.lower()
    verbs = BLOOM_VERBS.get(level, ["Explain", "Discuss", "Elaborate on"])
    return random.choice(verbs)


def _extract_question_from_audit(audit_text: str, drafted_question: str) -> str:
    """Extract clean question text from an auditor response, discarding meta commentary."""
    if not audit_text:
        return drafted_question

    text = audit_text.strip()

    # Pattern 1: Look for "Revised Question: ..." or "Question: ..."
    match_rev = re.search(r"(?i)(?:revised|confirmed|audited)?\s*question\s*[:\-]\s*(.+)", text, re.DOTALL)
    if match_rev:
        candidate = match_rev.group(1).strip()
        # Take first sentence or paragraph if multiline
        cand_lines = [l.strip() for l in candidate.split("\n") if l.strip()]
        if cand_lines:
            cand = cand_lines[0].strip(' "\'')
            if len(cand.split()) >= 6:
                return cand

    # Pattern 2: Look for quoted text inside the response
    quotes = re.findall(r'"([^"\n]{25,})"', text)
    if quotes:
        for q_cand in quotes:
            q_cand = q_cand.strip()
            # If it starts with any valid action verb or has question structure
            words = q_cand.split()
            if len(words) >= 6 and not q_cand.lower().startswith("the drafted"):
                return q_cand

    # Pattern 3: Find any line that starts with an action verb
    all_verbs = [v.lower() for v_list in BLOOM_VERBS.values() for v in v_list]
    for line in text.split("\n"):
        line_clean = line.strip().strip('"\'')
        first_word = line_clean.split()[0].lower() if line_clean.split() else ""
        if any(first_word == v.split()[0] for v in all_verbs) and len(line_clean.split()) >= 8:
            return line_clean

    # If the auditor output is purely commentary (e.g. "The drafted question looks good..."), retain original draft
    if any(text.lower().startswith(p) for p in ["the drafted", "after reviewing", "based on", "i confirm", "the question"]):
        return drafted_question

    return text


def clean_question_text(q: str) -> str:
    """Sanitize generated question text to remove introductory or meta-prompt chatter."""
    if not q:
        return ""
    q = re.sub(r"(?i)^(?:the\s+drafted\s+exam\s+question|after\s+reviewing|based\s+on\s+the\s+audit|here\s+is\s+(?:a|the|your)?)\s*.*?:", "", q.strip())
    q = re.sub(r"(?i)this\s+question.*", "", q)
    q = re.sub(r"\(.*?marks?\)", "", q, flags=re.I)
    q = re.sub(r"\[.*?\]", "", q)
    q = re.sub(r"^(?:Question\s*\d*[:.-]?|\d+[\.\):-]?)\s*", "", q.strip())
    q = q.replace("**", "").replace("*", "")
    q = q.strip().strip('"').strip("'")
    q = re.sub(r"\s+", " ", q)
    return q.strip()


def _call_llm_with_retry(
    llm: Any,
    prompt: str,
    max_retries: int = 2
) -> str:
    """Execute Ollama generation with exponential backoff retry and clear error messages."""
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.generate([prompt])
            text = response.generations[0][0].text.strip()
            if text:
                return text
        except Exception as e:
            last_exception = e
            print(f"[rag_engine] Ollama call attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(1)

    error_msg = (
        f"Ollama generation failed after {max_retries} attempts. "
        "Please ensure the Ollama service is running (`ollama serve`) and the target model is installed. "
        f"Original error: {last_exception}"
    )
    raise RuntimeError(error_msg)


def generate_exam_question(
    subtopic: Any,
    bloom_level: str,
    question_type: str,
    difficulty: str,
    kb: Optional[Any] = None,
    mode: str = "rag",
    prompt_style: str = "few_shot",
    model_name: str = "llama3",
    k: int = 3
) -> Dict[str, Any]:
    """Generate a calibrated 10-mark university exam question using RAG or Baseline approaches.

    Args:
        subtopic: Name or string description of syllabus concept.
        bloom_level: Target cognitive level ('remember', 'understand', 'apply', 'analyze', 'evaluate', 'create').
        question_type: 'theory' or 'numerical'.
        difficulty: 'Easy', 'Moderate', or 'Hard'.
        kb: Optional KnowledgeBase instance for RAG retrieval.
        mode: 'rag' (retrieves context from kb) or 'baseline' (zero retrieval).
        prompt_style: 'zero_shot', 'few_shot', or 'cot_check'.
        model_name: Local Ollama model name ('llama3', 'mistral', 'phi3', etc.).
        k: Number of reference chunks to retrieve in RAG mode.

    Returns:
        Dictionary containing:
        {
            "question": str,
            "retrieved_chunks": List[str],
            "prompt_used": str,
            "latency_s": float,
            "mode": str,
            "prompt_style": str
        }
    """
    if isinstance(subtopic, list):
        subtopic = ", ".join(str(s) for s in subtopic)
    subtopic_str = str(subtopic).strip()

    t_start = time.time()
    verb = _select_action_verb(bloom_level)
    diff_lower = difficulty.lower()

    # 1. Retrieval Phase
    retrieved_chunks: List[str] = []
    context_text = ""
    if mode.lower() == "rag" and kb is not None:
        try:
            results = kb.retrieve(subtopic_str, k=k)
            retrieved_chunks = [r[0] for r in results]
            if retrieved_chunks:
                context_text = "\n\n---\n\n".join(retrieved_chunks)
        except Exception as e:
            print(f"[rag_engine] Retrieval error: {e}. Falling back to zero context.")

    # 2. Build Prompt based on prompt_style and mode
    diff_instructions = {
        "Easy": "Use straightforward, fundamental terminology. Ask for essential definitions and working principles.",
        "Moderate": "Ask for internal mechanisms, comparative analysis, or logical derivations.",
        "Hard": "Frame complex real-world scenarios, critical trade-off evaluations, or multi-step formulations."
    }.get(difficulty, "Ensure comprehensive 10-mark depth.")

    # Base requirements
    type_instruction = (
        "Frame a numerical problem with concrete parameter values, requiring mathematical derivation and final value calculation."
        if question_type == "numerical"
        else "Ensure the question requires in-depth conceptual explanation, architectural diagrams, and practical engineering relevance."
    )

    system_persona = (
        "You are an expert Professor at B.M.S. College of Engineering (BMSCE), Department of AI & ML. "
        "You are framing a high-quality 10-mark question for a Semester End Main Examination."
    )

    # Few-shot section
    few_shot_block = ""
    if prompt_style == "few_shot":
        examples = _load_few_shot_examples(bloom_level)
        formatted_ex = []
        for idx, ex in enumerate(examples, start=1):
            formatted_ex.append(f"Example {idx}:\nSubtopic: {ex.get('subtopic')}\nQuestion: {ex.get('question')}")
        few_shot_block = "EXEMPLAR QUESTIONS (Learn the style and rigor from these):\n" + "\n\n".join(formatted_ex) + "\n\n"

    # Context injection block for RAG
    rag_context_block = ""
    if mode.lower() == "rag" and context_text:
        rag_context_block = f"""
SOURCE CONTEXT (Extracted from official curriculum textbooks and syllabus):
\"\"\"
{context_text}
\"\"\"

STRICT GROUNDING INSTRUCTIONS:
1. Ground the question strictly on the concepts, formulas, architectures, and mechanisms provided in the SOURCE CONTEXT above.
2. Do not invent ungrounded terminology or facts outside the curriculum context.
"""

    prompt = f"""{system_persona}

{rag_context_block}{few_shot_block}TASK SPECIFICATIONS:
- SUBTOPIC: {subtopic_str}
- TARGET COGNITIVE LEVEL (Bloom's Taxonomy): {bloom_level.upper()}
- QUESTION TYPE: {question_type.upper()} ({type_instruction})
- DIFFICULTY: {difficulty} ({diff_instructions})

MANDATORY RULES:
1. Start the question with the Bloom action verb stem: "{verb}".
2. Frame ONE coherent, comprehensive 10-mark examination question.
3. If the topic involves system components or architectures, require a neat diagram.
4. Output ONLY the question text. Do NOT include markdown bold markers (**), conversational greetings, or preamble like 'Here is your question'.

OUTPUT QUESTION TEXT ONLY:
"""

    # 3. LLM Generation
    llm = _get_llm(model_name)
    raw_question = ""
    final_prompt = prompt

    if llm is not None:
        try:
            raw_question = _call_llm_with_retry(llm, prompt, max_retries=2)

            # 4. Chain-of-Thought / Verification Step if requested
            if prompt_style == "cot_check" and raw_question:
                drafted_q = raw_question
                cot_prompt = f"""You are a Senior Academic Auditor for BMSCE examinations.
Review this drafted exam question:
"{raw_question}"

AUDIT SPECIFICATIONS:
- Target Subtopic: {subtopic_str}
- Target Bloom Level: {bloom_level.upper()} (Action verbs: {', '.join(BLOOM_VERBS.get(bloom_level.lower(), []))})
- Difficulty: {difficulty}

INSTRUCTIONS:
Verify if the question strictly meets the Bloom level '{bloom_level.upper()}' and covers '{subtopic_str}'.
- If it aligns, return the drafted question verbatim.
- If it does not align, revise it so it starts with an appropriate Bloom verb like '{verb}' and matches the target level.

CRITICAL CONSTRAINT: Output ONLY the question text itself. Do NOT include explanatory preamble, greetings, or phrases like 'The drafted question...'.

QUESTION TEXT ONLY:
"""
                revised = _call_llm_with_retry(llm, cot_prompt, max_retries=1)
                if revised:
                    raw_question = _extract_question_from_audit(revised, drafted_q)
                    final_prompt += f"\n\n--- CoT Verification Audit Prompt ---\n{cot_prompt}"

        except Exception as e:
            print(f"[rag_engine] LLM generation error: {e}. Utilizing fallback generator.")
            raw_question = _generate_rule_based_fallback(subtopic_str, bloom_level, question_type, difficulty, verb)
    else:
        raw_question = _generate_rule_based_fallback(subtopic_str, bloom_level, question_type, difficulty, verb)

    cleaned_q = clean_question_text(raw_question)
    latency = round(time.time() - t_start, 3)

    return {
        "question": cleaned_q,
        "retrieved_chunks": retrieved_chunks,
        "prompt_used": final_prompt,
        "latency_s": latency,
        "mode": mode,
        "prompt_style": prompt_style
    }


def _generate_rule_based_fallback(
    subtopic: str,
    bloom_level: str,
    question_type: str,
    difficulty: str,
    verb: str
) -> str:
    """Deterministic, high-quality fallback question generator if Ollama is unreachable."""
    if question_type == "numerical":
        return (
            f"{verb} a numerical problem on {subtopic}. "
            f"Given standard {difficulty.lower()}-level system parameters, calculate the required metrics, "
            "illustrate each intermediate derivation step, and analyze the resulting efficiency."
        )
    return (
        f"{verb} {subtopic} in detail. "
        "Illustrate the underlying mechanism with a neat block diagram, explain its core mathematical or logical foundations, "
        "and discuss its practical engineering trade-offs in modern computing systems."
    )
