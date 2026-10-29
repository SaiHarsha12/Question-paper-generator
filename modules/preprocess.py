"""Text Preprocessing and OCR Extraction Pipeline for LLM / NLP Workflows.

Course: Natural Language Processing for LLM (24AM7PCLLM)
Institution: B.M.S. College of Engineering (BMSCE)
Department: Artificial Intelligence & Machine Learning

This module implements a complete, academic-grade text preprocessing pipeline:
1. Multi-modal PDF text extraction (digital text layer + OCR fallback).
2. Text normalization and cleaning (Unicode, hyphenation, headers/footers).
3. Tokenization, stopword removal, lemmatization, and stemming comparison.
4. Sentence-aware text chunking with configurable overlap.
5. Quantitative preprocessing analytics for experimental reporting.
"""

from typing import List, Dict, Any, Optional, Set, Union
import os
import re
import unicodedata

import pdfplumber

# NLTK imports with safe fallbacks
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer

    # Ensure required NLTK resources are available without blocking on network if present
    for corpus_path, corpus_name in [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4")
    ]:
        try:
            nltk.data.find(corpus_path)
        except LookupError:
            try:
                nltk.download(corpus_name, quiet=True)
            except Exception:
                pass
except ImportError:
    nltk = None


def extract_text(pdf_path: str, return_pages: bool = False) -> Union[str, List[Dict[str, Any]]]:
    """Extract text from a PDF document using pdfplumber with OCR fallback.

    If a page contains no extractable text layer (e.g. scanned document or image),
    the function falls back to OCR via pdf2image and pytesseract.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        return_pages: If True, returns a list of dictionaries with page numbers
            and text. If False, returns a single concatenated string.

    Returns:
        Concatenated text string or a list of dictionaries per page:
        [{"page": int, "text": str, "method": "text_layer" | "ocr"}].
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    pages_data: List[Dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            method = "text_layer"

            # Check if text layer is empty or sparse (e.g. scanned image page)
            if not page_text.strip():
                ocr_text = _extract_page_ocr(pdf_path, idx)
                if ocr_text.strip():
                    page_text = ocr_text
                    method = "ocr"

            pages_data.append({
                "page": idx,
                "text": page_text,
                "method": method
            })

    if return_pages:
        return pages_data

    return "\n\n".join(p["text"] for p in pages_data if p["text"])


def _extract_page_ocr(pdf_path: str, page_number: int) -> str:
    """Helper function to perform OCR on a single PDF page."""
    try:
        from pdf2image import convert_from_path
        import pytesseract

        images = convert_from_path(
            pdf_path,
            first_page=page_number,
            last_page=page_number,
            dpi=200
        )
        if images:
            return pytesseract.image_to_string(images[0])
    except Exception as e:
        # OCR might fail if tesseract binary or poppler is not on system PATH
        print(f"[preprocess:OCR] Notice: OCR fallback skipped for page {page_number} ({e})")
    return ""


def clean_text(text: str) -> str:
    """Clean and normalize raw text using regex and Unicode normalization.

    Cleaning steps include:
    - Unicode NFKD normalization and standard ASCII character mapping.
    - Joining hyphenated line breaks (e.g., 'arti-\\nficial' -> 'artificial').
    - Removal of page number lines and running headers/footers.
    - Normalization of irregular whitespace while preserving paragraph boundaries.

    Args:
        text: Raw text string.

    Returns:
        Cleaned and normalized text string.
    """
    if not text:
        return ""

    # 1. Unicode normalization (NFKD)
    normalized = unicodedata.normalize("NFKD", text)

    # Standardize curly quotes, dashes, bullets
    char_map = {
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "—": "-", "–": "-", "•": "-", "…": "...",
        "\u00a0": " ", "\ufeff": ""
    }
    for k, v in char_map.items():
        normalized = normalized.replace(k, v)

    # 2. Join hyphenated words across line breaks (e.g., 'algo-\nrithm' -> 'algorithm')
    normalized = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", normalized)

    # 3. Remove running page numbers (e.g., 'Page 1 of 10', 'Page 3', standalone numbers)
    normalized = re.sub(r"(?i)\bPage\s+\d+\s*(?:of\s*\d+)?\b", "", normalized)
    normalized = re.sub(r"^\s*\d+\s*$", "", normalized, flags=re.MULTILINE)

    # 4. Remove generic running header/footer artifacts like semester timestamps
    normalized = re.sub(r"(?i)\bAcademic\s+Year\s+\d{4}[-\s]\d{2,4}\b", "", normalized)

    # 5. Collapse duplicate carriage returns and tabs
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n\s*\n\s*\n+", "\n\n", normalized)

    return normalized.strip()


def tokenize(text: str, lower: bool = True) -> List[str]:
    """Tokenize text into linguistic word tokens.

    Args:
        text: Input text string.
        lower: If True, converts all tokens to lowercase.

    Returns:
        List of string tokens.
    """
    if not text:
        return []

    processed = text.lower() if lower else text

    if nltk is not None:
        try:
            tokens = word_tokenize(processed)
            # Filter tokens to only alphanumeric words
            return [t for t in tokens if re.search(r"\w", t)]
        except Exception:
            pass

    # Regex-based fallback tokenizer
    return re.findall(r"\b\w+(?:'\w+)?\b", processed)


def remove_stopwords(tokens: List[str], custom_stopwords: Optional[Set[str]] = None) -> List[str]:
    """Remove English stopwords and punctuation tokens from a token list.

    Args:
        tokens: List of word tokens.
        custom_stopwords: Optional set of additional stopwords to remove.

    Returns:
        Filtered list of tokens without stopwords.
    """
    stop_words: Set[str] = set()
    if nltk is not None:
        try:
            stop_words = set(stopwords.words("english"))
        except Exception:
            stop_words = set()

    # Academic/administrative stop words commonly found in course syllabi
    default_domain_stops = {
        "etc", "hrs", "hours", "unit", "module", "chapter", "syllabus",
        "semester", "credits", "cie", "see", "marks", "course"
    }
    stop_words.update(default_domain_stops)

    if custom_stopwords:
        stop_words.update(s.lower() for s in custom_stopwords)

    return [t for t in tokens if t.lower() not in stop_words and len(t) > 1]


def lemmatize(tokens: List[str]) -> List[str]:
    """Lemmatize tokens to their base linguistic dictionary form using WordNet.

    Args:
        tokens: List of word tokens.

    Returns:
        List of lemmatized tokens.
    """
    if not tokens:
        return []

    if nltk is not None:
        try:
            lemmatizer = WordNetLemmatizer()
            # Lemmatize both noun and verb forms for comprehensive coverage
            result: List[str] = []
            for t in tokens:
                lemma = lemmatizer.lemmatize(t, pos="v")
                if lemma == t:
                    lemma = lemmatizer.lemmatize(t, pos="n")
                result.append(lemma)
            return result
        except Exception:
            pass

    return tokens


def stem(tokens: List[str], algorithm: str = "porter") -> List[str]:
    """Stem tokens to their morphological root for comparison with lemmatization.

    Args:
        tokens: List of word tokens.
        algorithm: 'porter' (default) or 'snowball'.

    Returns:
        List of stemmed tokens.
    """
    if not tokens:
        return []

    if nltk is not None:
        try:
            stemmer = SnowballStemmer("english") if algorithm == "snowball" else PorterStemmer()
            return [stemmer.stem(t) for t in tokens]
        except Exception:
            pass

    return tokens


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    source: str = "document",
    page: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Partition text into sentence-aware chunks with sliding window overlap.

    Unlike naive character chunking, this method respects sentence boundaries
    to preserve complete grammatical context for RAG embeddings.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum target character count per chunk.
        overlap: Character overlap between consecutive chunks.
        source: Name or identifier of the source document.
        page: Optional page number where text originated.

    Returns:
        List of chunk dicts:
        [{'chunk_id': str, 'text': str, 'source': str, 'page': Optional[int]}].
    """
    if not text or not text.strip():
        return []

    # Sentence boundary segmentation
    sentences: List[str] = []
    if nltk is not None:
        try:
            sentences = sent_tokenize(text)
        except Exception:
            pass

    if not sentences:
        # Regex sentence split fallback
        sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: List[Dict[str, Any]] = []
    current_sentences: List[str] = []
    current_len = 0
    chunk_index = 1

    for sentence in sentences:
        s_clean = sentence.strip()
        if not s_clean:
            continue

        s_len = len(s_clean)

        # If adding this sentence exceeds chunk_size and we already have content
        if current_len + s_len > chunk_size and current_sentences:
            chunk_content = " ".join(current_sentences)
            chunks.append({
                "chunk_id": f"{source}_chunk_{chunk_index}",
                "text": chunk_content,
                "source": source,
                "page": page
            })
            chunk_index += 1

            # Build overlapping history from trailing sentences
            overlap_sentences: List[str] = []
            overlap_len = 0
            for prev_s in reversed(current_sentences):
                if overlap_len + len(prev_s) <= overlap:
                    overlap_sentences.insert(0, prev_s)
                    overlap_len += len(prev_s)
                else:
                    break

            current_sentences = overlap_sentences + [s_clean]
            current_len = sum(len(s) for s in current_sentences) + len(current_sentences) - 1
        else:
            current_sentences.append(s_clean)
            current_len += s_len + 1

    # Flush remaining sentences
    if current_sentences:
        chunk_content = " ".join(current_sentences)
        chunks.append({
            "chunk_id": f"{source}_chunk_{chunk_index}",
            "text": chunk_content,
            "source": source,
            "page": page
        })

    return chunks


def preprocessing_report(text: str) -> Dict[str, Any]:
    """Generate comprehensive quantitative statistics of the preprocessing pipeline.

    Calculates:
    - Character counts (raw vs. cleaned).
    - Token counts before and after cleaning & stopword filtering.
    - Vocabulary size reductions.
    - Comparative examples of Stemming vs. Lemmatization.

    Args:
        text: Raw input text string.

    Returns:
        Dictionary of analytical metrics suitable for academic project reports.
    """
    raw_chars = len(text)
    cleaned = clean_text(text)
    cleaned_chars = len(cleaned)

    # Token analysis
    tokens_raw = tokenize(text)
    tokens_cleaned = tokenize(cleaned)
    tokens_no_stops = remove_stopwords(tokens_cleaned)

    # Vocabulary comparisons
    vocab_raw = set(tokens_cleaned)
    vocab_lemmatized = set(lemmatize(tokens_no_stops))
    vocab_stemmed = set(stem(tokens_no_stops))

    # Comparative samples illustrating stemming vs lemmatization
    comparison_samples: List[Dict[str, str]] = []
    test_words = ["running", "computational", "studies", "better", "networks", "engineering", "clustering"]
    sample_tokens = [w for w in test_words if w in vocab_raw] or test_words[:4]
    sample_lemmas = lemmatize(sample_tokens)
    sample_stems = stem(sample_tokens)

    for word, lm, stm in zip(sample_tokens, sample_lemmas, sample_stems):
        comparison_samples.append({
            "token": word,
            "lemmatized": lm,
            "stemmed": stm
        })

    chunks = chunk_text(cleaned, chunk_size=500, overlap=100)

    return {
        "raw_character_count": raw_chars,
        "cleaned_character_count": cleaned_chars,
        "compression_ratio": round(cleaned_chars / max(raw_chars, 1), 4),
        "tokens_before_cleaning": len(tokens_raw),
        "tokens_after_cleaning": len(tokens_cleaned),
        "tokens_after_stopwords": len(tokens_no_stops),
        "stopwords_removed": len(tokens_cleaned) - len(tokens_no_stops),
        "vocabulary_size_raw": len(vocab_raw),
        "vocabulary_size_lemmatized": len(vocab_lemmatized),
        "vocabulary_size_stemmed": len(vocab_stemmed),
        "stemming_vs_lemmatization_samples": comparison_samples,
        "total_chunks_generated": len(chunks)
    }

