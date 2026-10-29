import streamlit as st
import os, re, time, random
from modules.extractor import get_unit_concepts
from modules.rag_engine import generate_exam_question
from modules.pdf_builder import ExamPaperPDF

st.set_page_config(page_title="BMSCE Question Paper Generator", layout="wide")

st.markdown("""
    <style>
    /* Input Section Styling */
    .stTextInput, .stSelectbox {
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
    </style>
    """, unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(BASE_DIR, "218-2188840_bms-college-of-engineering-bangalore-bms-college-of-engineering-logo.png")

with st.container():
    col_title, col_logo = st.columns([5, 1])
    with col_title:
        st.markdown("""
            <div style="padding: 10px 0;">
                <p style="color: #1E3A8A; font-size: 38px; font-weight: bold; margin: 0; line-height: 1.2;">BMSCE Question Paper Generator</p>
                <p style="color: #10B981; font-size: 18px; font-style: italic; margin: 5px 0 0 0;">Department of Artificial Intelligence & Machine Learning</p>
            </div>
        """, unsafe_allow_html=True)
    with col_logo:
        if os.path.exists(logo_path):
            st.image(logo_path, width=110)

def get_co_po_mapping(bloom, q_type):
    co, po = "CO1", "PO1"
    if bloom in ["understand", "apply"]:
        co = "CO1"
    elif bloom == "analyze" or q_type == "numerical":
        co = "CO2"
    elif bloom in ["evaluate", "create"]:
        co = "CO3"
        
    if q_type == "numerical":
        po = "PO2"
    elif bloom == "create":
        po = "PO3"
    elif bloom == "analyze":
        po = "PO4"
    else:
        po = "PO1"
    return co, po

def choose_bloom_level():
    return random.choice(["understand", "apply", "analyze", "evaluate", "create"])

def choose_question_type():
    return "numerical" if random.randint(1, 100) <= 30 else "theory"

def clean_question(q):
    q = re.sub(r"(?i)here.*?:", "", q)
    q = re.sub(r"(?i)this question.*", "", q)
    q = re.sub(r"\(.*?marks?\)", "", q)
    q = re.sub(r"\[.*?\]", "", q)
    q = re.sub(r"^(?:Question\s*\d*[:.-]?|\d+[\.\):-]?)\s*", "", q.strip())
    q = q.replace("**", "").replace("*", "")
    q = re.sub(r"\s+", " ", q)
    return q.strip()

if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = None

col1, col2, col3 = st.columns(3)
with col1:
    branch = st.text_input("Branch", value="Artificial Intelligence and Machine Learning")
    code = st.text_input("Course Code", value="", placeholder="e.g. 24AM4PCCNS")
with col2:
    course = st.text_input("Course Name", value="", placeholder="e.g. Computer Networks and Security")
    sem = st.text_input("Semester", value="", placeholder="e.g. 4")
with col3:
    difficulty = st.selectbox("Difficulty Level", ["Easy", "Moderate", "Hard"], index=1)

pdf_file = st.file_uploader("Upload Syllabus PDF", type="pdf")

if pdf_file:
    os.makedirs("data", exist_ok=True)
    path = f"data/{pdf_file.name}"
    with open(path, "wb") as f:
        f.write(pdf_file.read())

    units = get_unit_concepts(path)

    if units:
        st.success(f"Successfully extracted {len(units)} units from the syllabus.")
    else:
        st.error("Could not find distinct Units in the syllabus. Please ensure the PDF contains unit/module headings.")

    if st.button("Generate Question Paper"):
        if not units:
            st.error("Please upload a valid syllabus PDF with units before generating.")
        else:
            animation_placeholder = st.empty()
            with animation_placeholder.container():
                st.markdown("""
                    <div style="text-align: center; padding: 20px; border: 1px dashed #ccc; border-radius: 10px; background: #f9f9f9;">
                        <div style="font-size: 50px;">✏️</div>
                        <h3 style='color: #1E3A8A;'>Generating questions with AI... Please wait.</h3>
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

            for i, (u, subtopics_list) in enumerate(units.items(), 1):
                status_text.text(f"Processing {u} ({i}/{total_units})...")
                pdf.unit_header(f"UNIT - {i}")
                
                used_topics = []
                for choice_block in range(2): 
                    for j in range(2):
                        bloom = choose_bloom_level()
                        q_type = choose_question_type()
                        curr_co, curr_po = get_co_po_mapping(bloom, q_type)

                        available = [t for t in subtopics_list if t not in used_topics]
                        chosen_topic = random.choice(available if available else (subtopics_list or [u]))
                        used_topics.append(chosen_topic)

                        q_raw = generate_exam_question(chosen_topic, bloom, q_type, difficulty)
                        q = clean_question(q_raw)

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
