# System Architecture & Technical Specifications

**Course**: Natural Language Processing for LLM (24AM7PCLLM)  
**Institution**: B.M.S. College of Engineering (BMSCE), Bengaluru  
**Department**: Department of Artificial Intelligence & Machine Learning  

---

## 1. Architectural Philosophy & Design Principles

The BMSCE Autonomous Question Paper Generator is engineered as a modular, transparent, and reproducible Natural Language Processing and LLM pipeline. It addresses the pedagogical challenges of automated examination drafting:
1. **Strict Local Execution**: 100% of LLM inference, embedding generation, classification, and vector retrieval executes locally on CPU/local GPU using Ollama and open-weights models, guaranteeing complete data privacy and zero API subscription costs.
2. **Accreditation-Driven Design**: Every generated question is algorithmically mapped to National Board of Accreditation (NBA) Course Outcomes (CO) and Programme Outcomes (PO/PSO).
3. **Rigorous Quality Guardrails**: Multiple layers of semantic and rule-based filters audit factual groundedness, format compliance, and question redundancy prior to PDF typesetting.
4. **Comparative Empirical Foundation**: Alternative NLP techniques are implemented and benchmarked across every core subsystem (e.g., Dense FAISS vs. Sparse TF-IDF, 4 Bloom classifiers, and 4 prompting/retrieval architectures).

---

## 2. End-to-End Component Breakdown

```mermaid
flowchart LR
    PDF["Syllabus / Course Plan PDF"] --> PRE["Preprocessor & Extractor"]
    PRE --> CHUNKS["Text Chunks & COs"]
    CHUNKS --> FAISS[("FAISS Vector Index")]
    SUB["Target Concept & Bloom Level"] --> RET["Retriever Engine"]
    FAISS --> RET
    RET --> RAG["RAG Prompt Assembler"]
    RAG --> LLM["Ollama LLM (llama3)"]
    LLM --> AUDIT["Quality Guard & Duplicate Filter"]
    AUDIT --> CLASS["Bloom Classifier & CO Mapper"]
    CLASS --> BUILD["PDF Typesetter (fpdf2)"]
    BUILD --> OUT["Autonomous Exam Paper PDF"]
```

---

### Component 1: Multi-Modal Ingestion & Preprocessing Subsystem
* **Files**: [`modules/preprocess.py`](file:///c:/Users/saiha/Downloads/QuestionPaperGen/modules/preprocess.py), [`modules/extractor.py`](file:///c:/Users/saiha/Downloads/QuestionPaperGen/modules/extractor.py)

#### 1. Dual-Engine Document Extraction
* **Digital PDF Text Extraction**: Uses `pdfplumber` to extract native vector text layers page-by-page.
* **Scanned Image OCR Fallback**: If `pdfplumber` extracts fewer than 30 characters on any page (indicating a scanned physical photocopy), the subsystem automatically routes the page through `pdf2image` and `pytesseract` to extract OCR text.

#### 2. Hybrid Syllabus & Course Plan Parsing
* **Standard Heading Format**: Regex patterns identify `UNIT - [I|V|1-5]` or `MODULE [1-5]` headers and extract subtopic lists.
* **Table-Style Course Plans**: If explicit unit headings are absent, the parser extracts tabular grids (columns: *Week | Class | Unit | Topic*), cleans out review sessions, and aggregates topics by unit number.
* **Course Outcome (CO) Extraction**: Automatically scans for strings matching `CO\d+:\s*(.+)` or extracts institutional descriptions defined in `data/course_config.json`.

#### 3. Linguistic Preprocessing Pipeline
* **Text Normalization**: Strips running page numbers, header/footer artifacts, trailing hyphenated line breaks (`token-\nization` $\rightarrow$ `tokenization`), and normalizes Unicode quotation marks and dashes.
* **Sentence-Aware Chunking**: Segmented with character window length $W = 500$ and overlap $O = 100$, respecting sentence punctuation boundaries (`.`, `!`, `?`, `\n`) rather than arbitrary character slicing.
* **Tokenization & Morphological Analysis**: NLTK-powered word and sentence tokenization, stopword removal, and comparative Stemming (`PorterStemmer`) vs. Lemmatization (`WordNetLemmatizer`).

---

### Component 2: Dual-Strategy Information Retrieval Subsystem
* **File**: [`modules/retriever.py`](file:///c:/Users/saiha/Downloads/QuestionPaperGen/modules/retriever.py)

#### 1. Dense Semantic Vector Store (`KnowledgeBase`)
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense embeddings).
* **FAISS Index**: Employs `faiss.IndexFlatIP` (Exact Inner Product search). Vectors are $L_2$-normalized prior to indexing so that inner product computes exact Cosine Similarity:
  $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = \hat{u} \cdot \hat{v}$$
* **Persistence**: Serializes the FAISS index (`faiss.write_index`) and metadata chunks (`metadata.json`) to `data/index_store/` for rapid disk reloading.

#### 2. Sparse Lexical Vector Store (`TFIDFRetriever`)
* **Feature Representation**: Scikit-learn `TfidfVectorizer` utilizing sublinear TF scaling and English stopword filtration.
* **Scoring**: Computes cosine similarity between sparse query vectors and chunk matrices via `cosine_similarity`.

#### 3. Information Retrieval (IR) Benchmark Engine
Provides `compare_retrievers(queries, ground_truth, k=4)` computing:
* **Precision@k**: $\frac{|\text{Retrieved}_k \cap \text{Relevant}|}{k}$
* **Recall@k**: $\frac{|\text{Retrieved}_k \cap \text{Relevant}|}{|\text{Relevant}|}$
* **Mean Reciprocal Rank (MRR)**: $\frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$

---

### Component 3: Prompt Engineering & LLM Orchestration
* **File**: [`modules/rag_engine.py`](file:///c:/Users/saiha/Downloads/QuestionPaperGen/modules/rag_engine.py)

#### 1. Prompt Engineering Strategies
* **Baseline (Zero-Shot, No Retrieval)**: Prompts the LLM directly using parametric memory.
* **RAG + Zero-Shot**: Retrieves the top-$k$ syllabus passages and injects them under a `CONTEXT CHUNKS` block with strict grounding instructions.
* **RAG + Few-Shot**: Injects three realistic, curated BMSCE exam questions from [`data/few_shot_examples.json`](file:///c:/Users/saiha/Downloads/QuestionPaperGen/data/few_shot_examples.json) matching the targeted Bloom level to calibrate length, tone, and cognitive framing.
* **RAG + CoT Self-Verification Check**: A two-stage pipeline:
  1. *Stage 1*: The model drafts a candidate 10-mark examination question.
  2. *Stage 2*: A secondary auditor prompt audits the drafted question against Bloom's taxonomy action verbs and subtopic coverage, returning a confirmed or revised question.
  3. *Audit Sanitizer*: `_extract_question_from_audit` strips conversational auditor preambles to ensure 100% format compliance.

#### 2. Bloom Taxonomy Action Verb Stems
To enforce cognitive depth, questions are constrained to use validated pedagogical action verbs:
* **Remember**: *Define, List, State, Identify, Recall*
* **Understand**: *Explain, Describe, Discuss in detail, Elaborate on, Illustrate*
* **Apply**: *Solve, Demonstrate, Illustrate the mechanism of, Calculate, Apply*
* **Analyze**: *Compare, Differentiate between, Analyze the structural differences between, Distinguish*
* **Evaluate**: *Justify, Critique, Assess the limitations of, Evaluate the efficiency of*
* **Create**: *Design, Formulate, Construct, Devise an end-to-end*

---

### Component 4: Cognitive Level Classification Subsystem
* **Files**: [`modules/bloom_classifier.py`](file:///c:/Users/saiha/Downloads/QuestionPaperGen/modules/bloom_classifier.py), [`scripts/train_bloom.py`](file:///c:/Users/saiha/Downloads/QuestionPaperGen/scripts/train_bloom.py)

#### 1. Dataset & Class Distribution
[`data/bloom_dataset.csv`](file:///c:/Users/saiha/Downloads/QuestionPaperGen/data/bloom_dataset.csv) comprises **210 curated university examination questions** balanced across all 6 Bloom levels (35 samples per class) with an 80/20 stratified split (seed = 42).

#### 2. Evaluated Classifiers
1. **Rule-Based**: Keyword and verb matching against known Bloom taxonomies.
2. **TF-IDF + Logistic Regression**: Word-level n-gram features ($n \in [1, 2]$) paired with balanced multinomial logistic regression.
3. **Sentence-BERT + Logistic Regression (Selected Best Model)**: 384-dimensional dense semantic representations from `all-MiniLM-L6-v2` classifying subtle syntactic nuances.
4. **Zero-Shot LLM (Ollama)**: In-context zero-shot classification prompting `llama3`.

---

### Component 5: Accreditation & Outcome Mapping Subsystem
* **File**: [`modules/co_mapper.py`](file:///c:/Users/saiha/Downloads/QuestionPaperGen/modules/co_mapper.py)

#### 1. Institutional Course Outcomes (24AM7PCLLM)
* **CO1**: Apply concepts and techniques to process and interpret NLP data.
* **CO2**: Analyze language modeling and computational approaches to identify linguistic patterns.
* **CO3**: Evaluate LLM adaptation, fine-tuning, and alignment techniques.
* **CO4**: Design, develop and demonstrate real-world LLM-based solutions.

#### 2. Bayesian Prior-Blended Semantic Mapping
The subsystem computes cosine similarity between the generated question embedding $q$ and each Course Outcome description embedding $c_i$. To reflect BMSCE examination conventions, the score is blended with a cognitive prior matrix:
$$S(q, \text{CO}_i) = \alpha \cdot \cos(q, c_i) + (1 - \alpha) \cdot P(\text{CO}_i \mid \text{Bloom})$$
where $\alpha = 0.70$. The CO with maximum score is assigned, and associated Programme Outcomes (e.g., PO1, PO2, PO3, PO4, PSO3) are resolved.

---

### Component 6: Automated Quality & Grounding Guardrails
* **File**: [`modules/quality.py`](file:///c:/Users/saiha/Downloads/QuestionPaperGen/modules/quality.py)

Every candidate question must pass four automated validation gates:
1. **Format Compliance**: Confirms the question starts with a valid Bloom action verb, has a length between 8 and 80 words, contains zero conversational preamble (*"Here is..."*), and contains no hard-coded mark annotations.
2. **Pairwise Duplicate Detection**: Computes pairwise cosine similarity between the candidate question and all previously accepted questions in the paper. If similarity $\ge 0.85$, the candidate is rejected and regenerated (up to 2 retries).
3. **Factual Groundedness Score**:
   $$\text{Groundedness} = \frac{|\text{Lemmatized Content Words in Question} \cap \text{Context Vocabulary}|}{|\text{Lemmatized Content Words in Question}|}$$
4. **Semantic Context Relevance**: Cosine similarity between question embedding and the mean embedding of retrieved context chunks.

---

### Component 7: Document Typesetting & Compilation Subsystem
* **File**: [`modules/pdf_builder.py`](file:///c:/Users/saiha/Downloads/QuestionPaperGen/modules/pdf_builder.py)

* **Engine**: Built with modern `fpdf2` utilizing `new_x=XPos.LMARGIN, new_y=YPos.NEXT` positioning.
* **Dynamic Row Calculation**: Uses `multi_cell(..., dry_run=True, output="LINES")` to compute exact question cell heights, preventing table cell clipping.
* **BMSCE Autonomous Structure**:
  - Institutional header, VTU affiliation, course metadata, and examination instructions.
  - 5 Unit sections with standard autonomous choice blocks (Question 1 vs Question 2 separated by centered `OR` delimiters).
  - Explicit columns for **Q.No**, **Questions**, **CO**, **PO**, and **Marks**.

---

### Component 8: Empirical Evaluation Benchmark Subsystem
* **Files**: [`evaluation/run_evaluation.py`](file:///c:/Users/saiha/Downloads/QuestionPaperGen/evaluation/run_evaluation.py), [`scripts/log_observation.py`](file:///c:/Users/saiha/Downloads/QuestionPaperGen/scripts/log_observation.py)

* **Test Suite**: Evaluates 5 test cases (`TC1` through `TC5`) across all 4 architectures (20 trials).
* **Reporting**: Aggregates metrics to [`results/evaluation.csv`](file:///c:/Users/saiha/Downloads/QuestionPaperGen/results/evaluation.csv), writes [`results/evaluation_summary.md`](file:///c:/Users/saiha/Downloads/QuestionPaperGen/results/evaluation_summary.md), and generates 4 high-resolution Matplotlib figures into [`results/figures/`](file:///c:/Users/saiha/Downloads/QuestionPaperGen/results/figures/).
* **Empirical Observation Registry**: Logs qualitative findings into [`results/observations.csv`](file:///c:/Users/saiha/Downloads/QuestionPaperGen/results/observations.csv).

---

## 3. Data Flow Walkthrough (From Syllabus to Final PDF)

```
[Uploaded Syllabus PDF]
         │
         ▼
1. modules/extractor.py
   ├── Identifies Units (1 to 5)
   └── Extracts embedded Course Outcomes (CO1..CO4)
         │
         ▼
2. modules/preprocess.py
   ├── Strips line breaks, page numbers, headers/footers
   └── Generates overlapping 500-char sentence-aware chunks
         │
         ▼
3. modules/retriever.py
   ├── Encodes chunks with all-MiniLM-L6-v2
   └── Indexes chunks in FAISS IndexFlatIP
         │
         ▼
4. modules/rag_engine.py (Per Question: 4 per Unit)
   ├── Selects subtopic and target Bloom cognitive level
   ├── Retrieves top-k chunks from FAISS (Cosine Sim)
   ├── Injects into Few-Shot / CoT prompt template
   └── Invokes local Ollama LLM (llama3)
         │
         ▼
5. modules/quality.py
   ├── Verifies action verb stem, length (8-80 words), and format
   ├── Checks pairwise similarity (< 0.85); retries up to 2 times
   └── Calculates Groundedness and Context Relevance
         │
         ▼
6. modules/bloom_classifier.py
   └── Verifies generated stem matches requested Bloom level
         │
         ▼
7. modules/co_mapper.py
   ├── Embeds question and calculates CO cosine similarity
   └── Maps to Course Outcomes (CO1..CO4) and PO/PSO matrix
         │
         ▼
8. modules/pdf_builder.py
   ├── Formats header, instructions, and choice structure
   └── Outputs BMSCE_Question_Paper.pdf
```

---

## 4. Error Handling and Resiliency Strategies

| Subsystem | Potential Failure / Edge Case | Mitigation / Fallback Path |
| :--- | :--- | :--- |
| **Extractor** | PDF lacks digital text layer (scanned photocopy) | Automatically activates OCR fallback via `pdf2image` + `pytesseract`. |
| **Extractor** | Syllabus lacks "UNIT" headings (table course plan) | Parses table columns (*Week, Class, Unit, Topic*) and groups by Unit integer. |
| **RAG Engine** | Ollama daemon not running or model fails | Employs 2-attempt retry with exponential backoff; falls back to rule-based template. |
| **Quality** | LLM generates duplicate or out-of-format question | Automatic regeneration loop (up to 2 retries) selecting alternative subtopics. |
| **PDF Builder** | Multiline text exceeds standard table cell height | Calculates dry-run line count (`dry_run=True`) to adjust cell height dynamically. |
| **Classifier** | S-BERT model weights unavailable | Rule-based action-verb keyword matcher acts as instant fallback. |

