import streamlit as st
import os
import re
import time
import random
import glob
import json
import pandas as pd

from modules.extractor import get_unit_concepts
from modules.retriever import KnowledgeBase
from modules.rag_engine import generate_exam_question
from modules.pdf_builder import ExamPaperPDF
from modules.bloom_classifier import get_bloom_classifier
from modules.co_mapper import map_question_to_co, load_course_config
from modules.quality import (
    format_check,
    is_candidate_duplicate,
    relevance_score,
    groundedness_score,
    quality_report
)
from evaluation.run_evaluation import (
    run_evaluation,
    load_test_cases,
    TEST_CASES_PATH,
    CSV_PATH as EVAL_CSV_PATH,
    SUMMARY_MD_PATH as EVAL_SUMMARY_MD_PATH,
    FIGURES_DIR as EVAL_FIGURES_DIR
)
from scripts.log_observation import (
    load_observations,
    save_observation,
    OBSERVATIONS_PATH
)

st.set_page_config(
    page_title="BMSCE Question Paper Generator & NLP Evaluation",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
    <style>
    /* Input Section Styling */
    .stTextInput, .stSelectbox, .stMultiSelect {
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 8px;
    }
    .header-box {
        background-color: rgba(255, 255, 255, 0.05); 
        padding: 20px 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.08);
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(BASE_DIR, "assets", "logo.png")

# ==============================================================================
# SIDEBAR CONTROLS: RAG, PROMPT, MODEL, DIFFICULTY & KNOWLEDGE BASE
# ==============================================================================
with st.sidebar:
    if os.path.exists(logo_path):
        st.image(logo_path, width=160)

    st.title("⚙️ RAG Engine Controls")
    st.markdown("**Course**: 24AM7PCLLM | BMSCE")

    generation_mode = st.selectbox(
        "Generation Architecture",
        ["RAG (Context-Augmented Retrieval)", "Baseline (No Retrieval)"],
        index=0,
        help="'RAG' retrieves ground-truth concepts from the vector store. 'Baseline' prompts the LLM directly without retrieval."
    )

    prompt_style_ui = st.selectbox(
        "Prompt Engineering Strategy",
        ["Few-Shot (In-Context Exemplars)", "Zero-Shot (Direct Instruction)", "CoT Check (Two-Stage Self-Verification)"],
        index=0,
        help="'Few-shot' injects 3 Bloom exemplars. 'CoT Check' runs a secondary verification call to audit Bloom alignment."
    )

    selected_model = st.selectbox(
        "Local Ollama Model",
        ["llama3", "mistral", "phi3"],
        index=0,
        help="Configurable local open-weights LLM running via Ollama."
    )

    difficulty = st.selectbox(
        "Exam Difficulty Level",
        ["Easy", "Moderate", "Hard"],
        index=1,
        help="Target cognitive difficulty level for question formulation."
    )

    top_k = st.slider(
        "Top-k Retrieved Chunks",
        min_value=1,
        max_value=8,
        value=3,
        step=1,
        help="Number of context chunks to retrieve from the FAISS KnowledgeBase in RAG mode."
    )

    st.markdown("---")
    st.subheader("📚 Reference Knowledge Base")
    st.caption("Upload textbook / notes PDFs to index into FAISS (`data/reference/`)")
    ref_uploads = st.file_uploader(
        "Upload Reference Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Reference textbooks or unit notes to supplement syllabus retrieval."
    )

    if ref_uploads:
        os.makedirs("data/reference", exist_ok=True)
        saved_count = 0
        for ref_file in ref_uploads:
            ref_dest = os.path.join("data", "reference", ref_file.name)
            if not os.path.exists(ref_dest):
                with open(ref_dest, "wb") as f:
                    f.write(ref_file.read())
                saved_count += 1
        if saved_count > 0:
            st.success(f"Added {saved_count} reference document(s) to knowledge base.")
            if "kb" in st.session_state:
                del st.session_state["kb"]

    existing_refs = glob.glob("data/reference/*.pdf")
    st.info(f"Active Reference Docs: {len(existing_refs)} file(s) available")


# Main Banner
with st.container():
    col_title, col_logo = st.columns([5, 1])
    with col_title:
        st.markdown("""
            <div style="padding: 10px 0;">
                <p style="color: #1E3A8A; font-size: 34px; font-weight: bold; margin: 0; line-height: 1.2;">BMSCE Autonomous Question Paper Generator</p>
                <p style="color: #10B981; font-size: 16px; font-style: italic; margin: 5px 0 0 0;">Department of Artificial Intelligence & Machine Learning | Course: 24AM7PCLLM (Natural Language Processing for LLM)</p>
            </div>
        """, unsafe_allow_html=True)
    with col_logo:
        if os.path.exists(logo_path):
            st.image(logo_path, width=105)


def get_co_po_mapping(bloom, q_type="theory", question="", cos=None, matrix=None):
    """Maps question and Bloom level to Course Outcome (CO) and PO list."""
    if question:
        co, _, pos = map_question_to_co(question, bloom_level=bloom, cos=cos, co_po_matrix=matrix)
        return co, ", ".join(pos)
    
    # Fallback prior mapping if question text is not yet generated
    cfg = load_course_config()
    b_lower = str(bloom).lower()
    co = cfg.get("bloom_co_prior", {}).get(b_lower, "CO1")
    matrix_entry = cfg.get("co_po_matrix", {}).get(co, {})
    pos = [k for k in matrix_entry.keys() if k.startswith("PO")]
    return co, ", ".join(pos if pos else ["PO1"])


def choose_question_type():
    return "numerical" if random.randint(1, 100) <= 30 else "theory"


def clean_question(q):
    if not q:
        return ""
    q = re.sub(r"(?i)here.*?:", "", q)
    q = re.sub(r"(?i)this question.*", "", q)
    q = re.sub(r"\(.*?marks?\)", "", q)
    q = re.sub(r"\[.*?\]", "", q)
    q = re.sub(r"^(?:Question\s*\d*[:.-]?|\d+[\.\):-]?)\s*", "", q.strip())
    q = q.replace("**", "").replace("*", "")
    q = re.sub(r"\s+", " ", q)
    return q.strip()


# Session State Management
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = None
if "generation_logs" not in st.session_state:
    st.session_state.generation_logs = []
if "quality_report" not in st.session_state:
    st.session_state.quality_report = None
if "eval_status_msg" not in st.session_state:
    st.session_state.eval_status_msg = ""

# Load Course Configuration (24AM7PCLLM defaults, editable by user)
course_cfg = load_course_config()

# Tab Navigation: Generator vs Empirical Benchmark vs Observations Log
tab_generator, tab_evaluation, tab_observations = st.tabs([
    "📝 Question Paper Generator",
    "📊 Empirical Evaluation Benchmark",
    "📋 Empirical Observations Log"
])

# ==============================================================================
# TAB 1: QUESTION PAPER GENERATOR
# ==============================================================================
with tab_generator:
    # Course Metadata Inputs
    col1, col2 = st.columns(2)
    with col1:
        branch = st.text_input("Branch", value=course_cfg.get("branch", "Artificial Intelligence and Machine Learning"))
        code = st.text_input("Course Code", value=course_cfg.get("course_code", "24AM7PCLLM"), placeholder="e.g. 24AM7PCLLM")
    with col2:
        course = st.text_input("Course Name", value=course_cfg.get("course_name", "Natural Language Processing for LLM"), placeholder="e.g. Natural Language Processing for LLM")
        sem = st.text_input("Semester", value=course_cfg.get("semester", "VII"), placeholder="e.g. VII")

    # Cognitive Level & Active Pipeline Configuration
    with st.expander("🎯 Cognitive Level & Pipeline Configuration (Bloom Distribution)", expanded=False):
        st.markdown(f"**Active Mode**: `{generation_mode}` | **Prompt**: `{prompt_style_ui}` | **Model**: `{selected_model}` | **Top-k**: `{top_k}` | **Difficulty**: `{difficulty}`")
        bloom_dist_choice = st.selectbox(
            "Target Bloom Cognitive Level Distribution",
            [
                "Balanced Mix (All 6 Levels: Remember to Create)",
                "Standard Autonomous College Scheme (Understand, Apply, Analyze)",
                "Higher-Order Cognitive Focus (Analyze, Evaluate, Create)",
                "Foundational Technical Focus (Remember, Understand, Apply)",
                "Custom Cognitive Level Selection"
            ],
            index=0,
            help="Specifies which cognitive levels to target across generated exam questions."
        )

        # Resolve target Bloom distribution
        if bloom_dist_choice == "Custom Cognitive Level Selection":
            target_bloom_levels = st.multiselect(
                "Select Cognitive Levels to Sample From",
                ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"],
                default=["Understand", "Apply", "Analyze", "Evaluate"]
            )
            if not target_bloom_levels:
                target_bloom_levels = ["Understand", "Apply"]
        elif "Standard Autonomous" in bloom_dist_choice:
            target_bloom_levels = ["Understand", "Apply", "Analyze"]
        elif "Higher-Order" in bloom_dist_choice:
            target_bloom_levels = ["Analyze", "Evaluate", "Create"]
        elif "Foundational" in bloom_dist_choice:
            target_bloom_levels = ["Remember", "Understand", "Apply"]
        else:
            target_bloom_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

    pdf_file = st.file_uploader("Upload Syllabus PDF (Text-based, Scanned OCR, or Table-style Course Plan)", type="pdf")

    if pdf_file:
        os.makedirs("data", exist_ok=True)
        path = f"data/{pdf_file.name}"
        with open(path, "wb") as f:
            f.write(pdf_file.read())

        course_data = get_unit_concepts(path, return_full=True)
        units = course_data.get("units", {})
        extracted_cos = course_data.get("cos", {})

        # Determine active COs: Use extracted COs if present, else fallback to course_config.json
        if extracted_cos:
            active_cos = extracted_cos
            co_source = "Uploaded Syllabus PDF"
        else:
            active_cos = course_cfg.get("cos", {})
            co_source = "Default (data/course_config.json)"

        active_matrix = course_cfg.get("co_po_matrix", {})

        if units:
            st.success(f"Successfully extracted {len(units)} units from the syllabus.")
            st.info(f"Active Course Outcomes ({co_source}): {', '.join(active_cos.keys())}")
        else:
            st.error("Could not find distinct Units in the syllabus. Please ensure the PDF contains unit/module headings or a table course plan.")

        with st.expander("🎯 Course Outcomes (CO) & Accreditation Mapping (PO/PSO)", expanded=False):
            st.caption(f"Course Outcomes Source: {co_source}")
            co_summary = []
            for co_id, desc in active_cos.items():
                mapped_pos = list(active_matrix.get(co_id, {}).keys())
                po_display = ", ".join(mapped_pos) if mapped_pos else "PO1 (Default)"
                co_summary.append({"Outcome": co_id, "Description": desc, "Mapped PO/PSO": po_display})
            st.table(co_summary)

        # Initialize KnowledgeBase for RAG mode
        if "RAG" in generation_mode:
            if "kb" not in st.session_state or st.session_state.get("kb_source") != pdf_file.name:
                with st.spinner("Building FAISS Vector Index from syllabus & reference documents..."):
                    kb = KnowledgeBase()
                    reference_docs = [path] + glob.glob("data/reference/*.pdf")
                    kb.build(reference_docs)
                    st.session_state.kb = kb
                    st.session_state.kb_source = pdf_file.name
                    st.session_state.kb_chunks_count = len(kb.chunks)

            st.caption(f"Vector Store Active: {st.session_state.get('kb_chunks_count', 0)} chunks indexed (all-MiniLM-L6-v2 + FAISS IndexFlatIP).")

        if st.button("Generate Question Paper", type="primary"):
            if not units:
                st.error("Please upload a valid syllabus PDF with units before generating.")
            else:
                animation_placeholder = st.empty()
                with animation_placeholder.container():
                    st.markdown(f"""
                        <div style="text-align: center; padding: 20px; border: 1px dashed #ccc; border-radius: 10px; background: #f9f9f9;">
                            <div style="font-size: 50px;">✏️</div>
                            <h3 style='color: #1E3A8A;'>Generating questions with {selected_model} ({generation_mode.split()[0]})... Please wait.</h3>
                        </div>
                    """, unsafe_allow_html=True)

                progress_bar = st.progress(0.0)
                status_text = st.empty()

                pdf = ExamPaperPDF()
                pdf.add_page()
                display_code = code.strip() if code.strip() else "COURSE"
                display_course = course.strip() if course.strip() else "Course Name"
                display_sem = sem.strip() if sem.strip() else "N/A"
                pdf.draw_first_page_header(branch.strip(), display_code, display_course, display_sem)

                qno = 1
                total_units = len(units)
                total_questions = total_units * 4
                q_count = 0
                logs = []

                # Map UI selection to rag_engine arguments
                gen_mode = "rag" if "RAG" in generation_mode else "baseline"
                style_map = {
                    "Few-Shot (In-Context Exemplars)": "few_shot",
                    "Zero-Shot (Direct Instruction)": "zero_shot",
                    "CoT Check (Two-Stage Self-Verification)": "cot_check"
                }
                active_style = style_map.get(prompt_style_ui, "few_shot")
                active_kb = st.session_state.get("kb") if gen_mode == "rag" else None

                # Load classifier to verify questions post-generation
                bloom_verifier = get_bloom_classifier("best")

                for i, (u, subtopics_list) in enumerate(units.items(), 1):
                    status_text.text(f"Processing {u} ({i}/{total_units})...")
                    pdf.unit_header(f"UNIT - {i}")
                    
                    used_topics = []
                    for choice_block in range(2): 
                        for j in range(2):
                            target_bloom = random.choice(target_bloom_levels)
                            q_type = choose_question_type()

                            available = [t for t in subtopics_list if t not in used_topics]
                            chosen_topic = random.choice(available if available else (subtopics_list or [u]))

                            # Automatic Quality & Duplicate Guard: Regenerate up to 2 times if checks fail
                            max_retries = 2
                            attempt = 0
                            final_gen_output = None
                            final_q = ""
                            final_topic = chosen_topic
                            fmt_result = None

                            while attempt <= max_retries:
                                gen_output = generate_exam_question(
                                    final_topic,
                                    target_bloom.lower(),
                                    q_type,
                                    difficulty,
                                    kb=active_kb,
                                    mode=gen_mode,
                                    prompt_style=active_style,
                                    model_name=selected_model,
                                    k=top_k
                                )
                                candidate_q = clean_question(gen_output.get("question", ""))
                                fmt_result = format_check(candidate_q)
                                is_dup, max_dup_sim, _ = is_candidate_duplicate(
                                    candidate_q,
                                    [l["question"] for l in logs],
                                    threshold=0.85
                                )

                                final_gen_output = gen_output
                                final_q = candidate_q

                                # Accept if formatted properly and no duplicate
                                if fmt_result["valid"] and not is_dup:
                                    break

                                attempt += 1
                                if attempt <= max_retries:
                                    alt_pool = [t for t in subtopics_list if t != final_topic]
                                    final_topic = random.choice(alt_pool if alt_pool else (subtopics_list or [u]))

                            q = final_q
                            gen_output = final_gen_output
                            retrieved_chunks = gen_output.get("retrieved_chunks", [])
                            used_topics.append(final_topic)

                            # Semantic CO & PO Mapping using Sentence-Transformers + Bloom Prior
                            curr_co, co_sim, po_list = map_question_to_co(
                                question=q,
                                bloom_level=target_bloom,
                                cos=active_cos,
                                co_po_matrix=active_matrix,
                                blend_prior=True
                            )
                            curr_po = ", ".join(po_list)

                            # Post-generation Bloom verification via classifier
                            predicted_level = bloom_verifier.predict(q)
                            bloom_match = (predicted_level.strip().lower() == target_bloom.strip().lower())

                            # Quality & Grounding Evaluation
                            q_relevance = relevance_score(q, retrieved_chunks)
                            q_groundedness = groundedness_score(q, retrieved_chunks)

                            logs.append({
                                "qno": qno,
                                "unit": u,
                                "topic": final_topic,
                                "target_bloom": target_bloom,
                                "predicted_bloom": predicted_level,
                                "bloom_match": "✅ Aligned" if bloom_match else "⚠️ Mismatch",
                                "co": curr_co,
                                "co_sim": co_sim,
                                "po": curr_po,
                                "relevance": q_relevance,
                                "groundedness": q_groundedness,
                                "format_status": "✅ Pass" if fmt_result["valid"] else "⚠️ Issue",
                                "retries": attempt,
                                "type": q_type,
                                "latency_s": gen_output.get("latency_s", 0.0),
                                "retrieved_count": len(retrieved_chunks),
                                "retrieved_chunks": retrieved_chunks,
                                "question": q
                            })

                            pdf.question_row(str(qno), q, curr_co, curr_po, "10")
                            qno += 1
                            q_count += 1
                            progress_bar.progress(min(q_count / total_questions, 1.0))

                        if choice_block == 0:
                            pdf.or_row()

                clean_code = re.sub(r'[\\/*?:"<>| ]', '_', display_code)
                out = f"BMSCE_{clean_code}_Question_Paper.pdf"
                pdf.output(out)

                with open(out, "rb") as f:
                    st.session_state.pdf_data = f.read()
                st.session_state.pdf_filename = out
                st.session_state.generation_logs = logs
                st.session_state.quality_report = quality_report(logs)

                animation_placeholder.empty()
                progress_bar.empty()
                status_text.success("Question Paper Generated Successfully!")

    if st.session_state.pdf_data is not None:
        st.markdown("---")
        st.subheader("Generated Question Paper Ready")
        st.download_button(
            label="📥 Download Question Paper (PDF)",
            data=st.session_state.pdf_data,
            file_name=st.session_state.pdf_filename or "Question_Paper.pdf",
            mime="application/pdf",
            key="download_pdf_btn"
        )

        # Automated Quality & Grounding Audit Report Expander
        if st.session_state.get("quality_report"):
            q_rep = st.session_state.quality_report
            q_sum = q_rep.get("summary", {})
            with st.expander("🛡️ Question Paper Quality & Grounding Audit Report", expanded=True):
                qc1, qc2, qc3, qc4 = st.columns(4)
                with qc1:
                    st.metric("Format Compliance", q_sum.get("format_compliance_rate", "N/A"))
                with qc2:
                    st.metric("Avg Groundedness", f"{q_sum.get('avg_groundedness', 0.0) * 100:.1f}%")
                with qc3:
                    st.metric("Avg Context Relevance", f"{q_sum.get('avg_relevance', 0.0):.3f}")
                with qc4:
                    st.metric("Duplicates Flagged", q_sum.get("duplicate_count", 0))

                if q_sum.get("overall_passed"):
                    st.success(f"✅ Quality Audit Passed: All {q_sum.get('total_questions')} questions verified against BMSCE format guidelines with zero duplicates.")
                else:
                    st.info(f"ℹ️ Quality Audit Summary: Complete evaluation across {q_sum.get('total_questions')} questions. Duplicates flagged: {q_sum.get('duplicate_count')}.")

                if q_rep.get("per_question"):
                    df_quality = pd.DataFrame(q_rep["per_question"])
                    display_cols = [c for c in ["qno", "unit", "format_valid", "groundedness", "relevance", "duplicate_risk", "word_count", "question"] if c in df_quality.columns]
                    st.dataframe(df_quality[display_cols], use_container_width=True)

        # Detailed RAG & Bloom Classification Inspection Table
        if st.session_state.generation_logs:
            with st.expander("🔍 RAG Analytics & Bloom Level Verification Trace", expanded=False):
                total_qs = len(st.session_state.generation_logs)
                matches = sum(1 for l in st.session_state.generation_logs if "Aligned" in l["bloom_match"])
                avg_lat = sum(l['latency_s'] for l in st.session_state.generation_logs) / max(total_qs, 1)
                avg_co_sim = sum(l.get('co_sim', 0.0) for l in st.session_state.generation_logs) / max(total_qs, 1)

                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                with stat_col1:
                    st.metric("Total Questions", total_qs)
                with stat_col2:
                    st.metric("Bloom Alignment Rate", f"{(matches / total_qs) * 100:.1f}%", f"{matches}/{total_qs} matched")
                with stat_col3:
                    st.metric("Avg CO Semantic Sim", f"{avg_co_sim:.3f}")
                with stat_col4:
                    st.metric("Avg Latency", f"{avg_lat:.2f}s")

                clean_logs = [
                    {k: v for k, v in l.items() if k != "retrieved_chunks"}
                    for l in st.session_state.generation_logs
                ]
                st.dataframe(pd.DataFrame(clean_logs), use_container_width=True)


# ==============================================================================
# TAB 2: EMPIRICAL EVALUATION BENCHMARK
# ==============================================================================
with tab_evaluation:
    st.markdown("### 📊 Empirical Evaluation Benchmark: Architecture Comparison")
    st.markdown("""
    This benchmark rigorously evaluates and compares **four LLM prompting and retrieval architectures** 
    for academic exam paper generation across **five diverse test cases** on the **24AM7PCLLM** syllabus:
    - **Baseline (no retrieval, zero-shot)**: Direct LLM generation relying entirely on parametric memory.
    - **RAG + zero-shot**: Dense FAISS retrieval (`all-MiniLM-L6-v2`) injected into direct zero-shot prompt.
    - **RAG + few-shot**: Dense FAISS retrieval combined with calibrated in-context exemplars.
    - **RAG + cot_check**: Dense FAISS retrieval with two-stage Chain-of-Thought self-verification audit.
    """)

    st.markdown("---")
    eval_ctl1, eval_ctl2 = st.columns([3, 1])
    with eval_ctl1:
        eval_model_choice = st.selectbox(
            "Evaluation LLM Model (Local Ollama)",
            ["llama3", "mistral", "phi3"],
            index=0,
            key="eval_model_selector"
        )
    with eval_ctl2:
        st.write("")
        st.write("")
        run_benchmark_btn = st.button("▶️ Run Empirical Benchmark", type="primary", use_container_width=True)

    if run_benchmark_btn:
        eval_progress = st.progress(0.0)
        eval_status = st.empty()

        def update_eval_progress(pct: float, msg: str):
            eval_progress.progress(pct)
            eval_status.info(f"⚡ {msg}")

        try:
            with st.spinner("Executing empirical evaluation benchmark across test cases..."):
                results = run_evaluation(
                    model_name=eval_model_choice,
                    progress_callback=update_eval_progress
                )
            eval_progress.progress(1.0)
            eval_status.success("🎉 Empirical benchmark complete! Results, CSV, and figures updated.")
            st.session_state.eval_status_msg = "Completed successfully."
        except Exception as e:
            eval_status.error(f"Benchmark execution error: {e}")

    # Display Benchmark Results if results/evaluation.csv exists
    if os.path.exists(EVAL_CSV_PATH):
        st.markdown("#### 📈 Benchmark Summary Table")
        df_eval = pd.read_csv(EVAL_CSV_PATH)
        st.dataframe(df_eval, use_container_width=True)

        # Download Buttons
        dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 2])
        with dl_col1:
            with open(EVAL_CSV_PATH, "rb") as f:
                st.download_button(
                    label="📥 Download evaluation.csv",
                    data=f.read(),
                    file_name="evaluation.csv",
                    mime="text/csv",
                    key="dl_eval_csv"
                )
        with dl_col2:
            if os.path.exists(EVAL_SUMMARY_MD_PATH):
                with open(EVAL_SUMMARY_MD_PATH, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="📄 Download evaluation_summary.md",
                        data=f.read(),
                        file_name="evaluation_summary.md",
                        mime="text/markdown",
                        key="dl_eval_md"
                    )

        # Render 4 Matplotlib Figures
        fig1_path = os.path.join(EVAL_FIGURES_DIR, "relevance_vs_groundedness.png")
        fig2_path = os.path.join(EVAL_FIGURES_DIR, "bloom_alignment_comparison.png")
        fig3_path = os.path.join(EVAL_FIGURES_DIR, "latency_comparison.png")
        fig4_path = os.path.join(EVAL_FIGURES_DIR, "format_compliance_comparison.png")

        st.markdown("#### 📊 Evaluation Visualizations (4 Metrics)")
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            if os.path.exists(fig1_path):
                st.image(fig1_path, caption="Figure 1: Semantic Relevance vs. Groundedness by Architecture", use_container_width=True)
            if os.path.exists(fig3_path):
                st.image(fig3_path, caption="Figure 3: Mean Inference Latency Comparison (s)", use_container_width=True)

        with fig_col2:
            if os.path.exists(fig2_path):
                st.image(fig2_path, caption="Figure 2: Bloom Cognitive Level Alignment Accuracy (%)", use_container_width=True)
            if os.path.exists(fig4_path):
                st.image(fig4_path, caption="Figure 4: BMSCE Exam Format Compliance Rate (%)", use_container_width=True)

    # Academic Test Cases Viewer
    with st.expander("📂 View Academic Benchmark Test Cases (5 Defined Test Scenarios)", expanded=False):
        if os.path.exists(TEST_CASES_PATH):
            tc_data = load_test_cases(TEST_CASES_PATH)
            tc_rows = []
            for tc in tc_data:
                tc_rows.append({
                    "Test Case ID": tc.get("id"),
                    "Unit": tc.get("unit"),
                    "Subtopic": tc.get("subtopic"),
                    "Target Bloom": tc.get("bloom_level"),
                    "Difficulty": tc.get("difficulty"),
                    "Format": tc.get("source_format"),
                    "Ground Truth Excerpt": (tc.get("ground_truth_context", "")[:80] + "...") if tc.get("ground_truth_context") else "N/A"
                })
            st.dataframe(pd.DataFrame(tc_rows), use_container_width=True)

    # Full Markdown Report Viewer
    if os.path.exists(EVAL_SUMMARY_MD_PATH):
        with st.expander("📄 View Full Markdown Empirical Evaluation Report", expanded=False):
            with open(EVAL_SUMMARY_MD_PATH, "r", encoding="utf-8") as f:
                st.markdown(f.read())


# ==============================================================================
# TAB 3: EMPIRICAL OBSERVATIONS LOG
# ==============================================================================
with tab_observations:
    st.markdown("### 📋 Empirical Observations & Failure Modes Log")
    st.markdown("""
    **Course Requirement (24AM7PCLLM)**: Record at least **5 qualitative and empirical observations** 
    across at least **4 test cases** demonstrating hallucination mitigation, Bloom cognitive fidelity, 
    and prompt engineering behavior.
    """)

    # Metric Cards for Course Requirements Check
    all_obs = load_observations()
    total_obs = len(all_obs)
    tc_covered = len(set(o.get("test_case_id") for o in all_obs))

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Total Recorded Observations", total_obs, delta="Target: >= 5" if total_obs >= 5 else "Under Target")
    with m_col2:
        st.metric("Unique Test Cases Covered", tc_covered, delta="Target: >= 4" if tc_covered >= 4 else "Under Target")
    with m_col3:
        if total_obs >= 5 and tc_covered >= 4:
            st.success("✅ Academic Requirement Satisfied")
        else:
            st.warning("⚠️ Academic Requirement Pending")

    st.markdown("---")

    # Form to Log a New Manual Observation
    with st.expander("➕ Log New Manual Qualitative Observation", expanded=False):
        with st.form("new_observation_form", clear_on_submit=True):
            f_col1, f_col2, f_col3 = st.columns([1, 2, 1])
            with f_col1:
                tc_input = st.selectbox("Test Case ID", ["TC1", "TC2", "TC3", "TC4", "TC5", "Custom"])
                if tc_input == "Custom":
                    tc_id_val = st.text_input("Custom Test Case ID", value="TC-Custom")
                else:
                    tc_id_val = tc_input
            with f_col2:
                cfg_val = st.selectbox(
                    "Architecture Configuration",
                    [
                        "RAG + few-shot",
                        "RAG + zero-shot",
                        "RAG + cot_check",
                        "Baseline (no retrieval, zero-shot)"
                    ]
                )
            with f_col3:
                status_val = st.selectbox("Verdict Status", ["PASS", "FAIL"])

            exp_text = st.text_area(
                "Expected Pedagogical Behavior",
                placeholder="e.g. Question must strictly ground in Byte-Pair Encoding subword vocabulary merging rules without parametric hallucinations."
            )
            act_text = st.text_area(
                "Actual Model Output / Behavior",
                placeholder="e.g. Model retrieved BPE chunks from FAISS vector index and correctly framed a 10-mark comparison question."
            )
            rem_text = st.text_area(
                "Analytical Remarks / Failure Mode Notes",
                placeholder="e.g. Noticeable increase in groundedness score (0.79 vs 0.20 in baseline). Zero out-of-syllabus terms detected."
            )

            submit_obs_btn = st.form_submit_button("💾 Save Observation to results/observations.csv", type="primary")

            if submit_obs_btn:
                if not exp_text.strip() or not act_text.strip():
                    st.error("Please fill in both Expected Behavior and Actual Output before saving.")
                else:
                    new_obs_id = save_observation(
                        test_case_id=tc_id_val,
                        configuration=cfg_val,
                        expected_behavior=exp_text,
                        actual_output=act_text,
                        status=status_val,
                        remarks=rem_text
                    )
                    st.success(f"Observation {new_obs_id} saved to results/observations.csv!")
                    st.rerun()

    # View Recorded Observations Table
    if all_obs:
        st.markdown("#### 📑 Recorded Observations Registry")
        df_obs = pd.DataFrame(all_obs)
        st.dataframe(df_obs, use_container_width=True)

        # Download Observations CSV
        if os.path.exists(OBSERVATIONS_PATH):
            with open(OBSERVATIONS_PATH, "rb") as f:
                st.download_button(
                    label="📥 Download observations.csv",
                    data=f.read(),
                    file_name="observations.csv",
                    mime="text/csv",
                    key="dl_obs_csv"
                )
    else:
        st.info("No observations recorded yet. Use the form above to log observations.")
