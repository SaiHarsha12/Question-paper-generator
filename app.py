import streamlit as st
import os, re, time, random
from modules.extractor import get_unit_concepts
from modules.rag_engine import generate_exam_question
from modules.pdf_builder import ExamPaperPDF

st.set_page_config(page_title="BMSCE Question Paper Generator", layout="wide")

st.markdown("""
    <style>
    /* Main Container Shadow Box */
    .header-box {
        background-color: rgba(255, 255, 255, 0.05); 
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1); /* Subtle deep shadow */
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 30px;
        display: flex;
        align-items: center;
    }
    
    .title-container {
        text-align: center;
        width: 100%;
    }

    .main-title {
        color: #1E3A8A; /* Dark Blue */
        font-size: 42px !important;
        font-weight: bold;
        margin: 0px !important;
        padding: 0px !important;
        line-height: 1.2;
    }

    .sub-title {
        color: #10B981; /* Green */
        font-size: 20px !important;
        font-style: italic;
        margin: 5px 0 0 0 !important;
        padding: 0px !important;
    }

    /* Input Section Styling */
    .stTextInput, .stSelectbox {
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

logo_path = r"C:\Users\saiha\Downloads\QuestionPaperGen\218-2188840_bms-college-of-engineering-bangalore-bms-college-of-engineering-logo.png"

st.markdown('<div class="header-box">', unsafe_allow_html=True)

left_space, center_content, right_logo = st.columns([1, 6, 1.5])

with center_content:
    st.markdown("""
        <div class="title-container">
            <p class="main-title">BMSCE Question Paper Generator</p>
            <p class="sub-title">Department of Artificial Intelligence & Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)

with right_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=110)
    else:
        st.warning("Logo path not found")
st.markdown('</div>', unsafe_allow_html=True)

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
    q = re.sub(r"(?i)here.*?:","",q)
    q = re.sub(r"(?i)this question.*","",q)
    q = re.sub(r"\(.*?marks?\)","",q)
    q = re.sub(r"\s+"," ",q)
    return q.strip()

if "generate" not in st.session_state:
    st.session_state.generate = False

col1, col2, col3 = st.columns(3)
with col1:
    branch = st.text_input("Branch", "Artificial Intelligence and Machine Learning ")
    code = st.text_input("Course Code", " ")
with col2:
    course = st.text_input("Course Name", " ")
    sem = st.text_input("Semester", " ")
with col3:
    difficulty = st.selectbox("Difficulty Level", ["Easy", "Moderate", "Hard"])

pdf_file = st.file_uploader("Upload Syllabus PDF", type="pdf")

if pdf_file:
    os.makedirs("data", exist_ok=True)
    path = f"data/{pdf_file.name}"
    with open(path, "wb") as f:
        f.write(pdf_file.read())

    units = get_unit_concepts(path)

    if units:
        st.success(f"Successfully extracted {len(units)} units from the syllabus.")
    
    if st.button("Generate Question Paper"):
        st.session_state.generate = True

    if st.session_state.generate:
        animation_placeholder = st.empty()
        with animation_placeholder.container():
            st.markdown(f"""
                <div style="text-align: center; padding: 20px; border: 1px dashed #ccc; border-radius: 10px; background: #f9f9f9;">
                    <div style="font-size: 50px; animation: write 2s infinite linear;">✏️</div>
                    <h3 style='color: #1E3A8A;'>Questions are getting generated......</h3>
                </div>
                <style>
                    @keyframes write {{
                        0% {{ transform: translate(0,0) rotate(0deg); }}
                        25% {{ transform: translate(10px,-10px) rotate(10deg); }}
                        50% {{ transform: translate(20px,0) rotate(0deg); }}
                        75% {{ transform: translate(10px,10px) rotate(-10deg); }}
                        100% {{ transform: translate(0,0) rotate(0deg); }}
                    }}
                </style>
            """, unsafe_allow_html=True)
            time.sleep(5) 

        progress_bar = st.progress(0)
        status_text = st.empty()
        pdf = ExamPaperPDF()
        pdf.add_page()
        pdf.draw_first_page_header(branch, code, course, sem)

        qno = 1
        total_units = len(units)

        for i, (u, subtopics_list) in enumerate(units.items(), 1):
            status_text.text(f"Processing {u}...")
            pdf.unit_header(f"UNIT - {i}")
            
            used_topics = []
            for choice_block in range(2): 
                for j in range(2):
                    bloom = choose_bloom_level()
                    q_type = choose_question_type()
                    curr_co, curr_po = get_co_po_mapping(bloom, q_type)

                    available = [t for t in subtopics_list if t not in used_topics]
                    chosen_topic = random.choice(available if available else subtopics_list)
                    used_topics.append(chosen_topic)

                    q_raw = generate_exam_question(chosen_topic, bloom, q_type, difficulty)
                    q = clean_question(q_raw)

                    pdf.question_row(str(qno), q, curr_co, curr_po, "10")
                    qno += 1

                if choice_block == 0:
                    pdf.or_row()

                progress_percent = ((i-1) * 4 + (choice_block * 2 + j + 1)) / (total_units * 4)
                progress_bar.progress(min(progress_percent, 1.0))

        out = f"BMSCE_{code}_Question_Paper.pdf"
        pdf.output(out)
        animation_placeholder.empty()
        progress_bar.empty()
        status_text.success("Question Paper Generated Successfully!")

        with open(out, "rb") as f:
            st.download_button(label="📥 Download Question Paper (PDF)", data=f, file_name=out, mime="application/pdf")
        
        st.session_state.generate = False