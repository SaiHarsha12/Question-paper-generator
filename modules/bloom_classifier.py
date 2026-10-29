"""Bloom's Taxonomy Cognitive Level Classifiers.

Course: Natural Language Processing for LLM (24AM7PCLLM)
Institution: B.M.S. College of Engineering (BMSCE)
Department: Artificial Intelligence & Machine Learning

This module implements four distinct classification approaches behind a unified interface:
1. RuleBasedBloomClassifier: Action-verb keyword heuristics.
2. TFIDFBloomClassifier: Scikit-learn TF-IDF n-grams + Logistic Regression.
3. EmbeddingBloomClassifier: Sentence-BERT ('all-MiniLM-L6-v2') dense embeddings + Logistic Regression.
4. ZeroShotLLMBloomClassifier: In-context zero-shot classification via local Ollama LLM.

Unified Interface:
    predict(question: str) -> str
"""

from typing import List, Dict, Any, Optional, Union
import os
import re
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    OllamaLLM = None

from modules.preprocess import clean_text

BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

# Action verb stems and pedagogical keyword indicators for Rule-Based classifier
BLOOM_KEYWORD_MAP: Dict[str, List[str]] = {
    "Remember": [
        "define", "list", "state", "identify", "recall", "name", "mention",
        "which of", "what is", "who", "when", "where", "write short note on"
    ],
    "Understand": [
        "explain", "describe", "discuss", "elaborate", "illustrate the working",
        "summarize", "interpret", "clarify", "paraphrase", "give reasons"
    ],
    "Apply": [
        "solve", "calculate", "compute", "demonstrate", "apply", "derive",
        "determine", "implement", "trace", "execute", "find the value", "simulate"
    ],
    "Analyze": [
        "compare", "contrast", "differentiate", "distinguish", "analyze",
        "categorize", "classify the differences", "break down", "examine the trade-offs", "vs"
    ],
    "Evaluate": [
        "justify", "critique", "assess", "evaluate", "appraise", "judge",
        "defend", "validate", "rate", "prioritize", "criticize", "argue whether"
    ],
    "Create": [
        "design", "formulate", "construct", "devise", "develop", "compose",
        "synthesize", "propose a new", "architect", "invent", "plan a system"
    ]
}

MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "bloom_classifier_model.joblib"
)


class RuleBasedBloomClassifier:
    """Heuristic rule-based classifier utilizing Bloom's Taxonomy action verbs."""

    def __init__(self):
        self.levels = BLOOM_LEVELS

    def predict(self, question: str) -> str:
        """Classify a question by matching initial action verbs and keyword frequencies."""
        if not question:
            return "Understand"

        q_clean = clean_text(question).lower()
        first_few_words = " ".join(q_clean.split()[:4])

        # Priority 1: Check leading action verb stem
        for level, keywords in BLOOM_KEYWORD_MAP.items():
            for kw in keywords:
                if first_few_words.startswith(kw) or re.search(rf"\b{re.escape(kw)}\b", first_few_words):
                    return level

        # Priority 2: Overall frequency match across entire question
        level_scores: Dict[str, int] = {lvl: 0 for lvl in BLOOM_LEVELS}
        for level, keywords in BLOOM_KEYWORD_MAP.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", q_clean):
                    level_scores[level] += 1

        best_level = max(level_scores, key=level_scores.get)
        if level_scores[best_level] > 0:
            return best_level

        # Fallback default
        return "Understand"

    def predict_batch(self, questions: List[str]) -> List[str]:
        return [self.predict(q) for q in questions]


class TFIDFBloomClassifier:
    """Statistical NLP Classifier using TF-IDF sublinear n-grams and Logistic Regression."""

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=3000,
                sublinear_tf=True,
                stop_words="english"
            )),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ])
        self.is_fitted = False

    def fit(self, X: List[str], y: List[str]) -> "TFIDFBloomClassifier":
        cleaned_X = [clean_text(x) for x in X]
        self.pipeline.fit(cleaned_X, y)
        self.is_fitted = True
        return self

    def predict(self, question: str) -> str:
        if not self.is_fitted:
            raise RuntimeError("TFIDFBloomClassifier must be trained with .fit() before predicting.")
        pred = self.pipeline.predict([clean_text(question)])[0]
        return str(pred)

    def predict_batch(self, questions: List[str]) -> List[str]:
        if not self.is_fitted:
            raise RuntimeError("TFIDFBloomClassifier must be trained with .fit() before predicting.")
        cleaned = [clean_text(q) for q in questions]
        return [str(p) for p in self.pipeline.predict(cleaned)]


class EmbeddingBloomClassifier:
    """Dense Semantic Classifier using Sentence-Transformers (all-MiniLM-L6-v2) + Logistic Regression."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        self.is_fitted = False

    def fit(self, X: List[str], y: List[str]) -> "EmbeddingBloomClassifier":
        cleaned_X = [clean_text(x) for x in X]
        embeddings = self.encoder.encode(
            cleaned_X,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        self.clf.fit(embeddings, y)
        self.is_fitted = True
        return self

    def predict(self, question: str) -> str:
        if not self.is_fitted:
            raise RuntimeError("EmbeddingBloomClassifier must be trained with .fit() before predicting.")
        emb = self.encoder.encode(
            [clean_text(question)],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        pred = self.clf.predict(emb)[0]
        return str(pred)

    def predict_batch(self, questions: List[str]) -> List[str]:
        if not self.is_fitted:
            raise RuntimeError("EmbeddingBloomClassifier must be trained with .fit() before predicting.")
        cleaned = [clean_text(q) for q in questions]
        embeddings = self.encoder.encode(
            cleaned,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return [str(p) for p in self.clf.predict(embeddings)]


class ZeroShotLLMBloomClassifier:
    """Zero-shot LLM prompting classifier using local Ollama model."""

    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.llm = OllamaLLM(model=model_name, streaming=False) if OllamaLLM else None

    def predict(self, question: str) -> str:
        if not self.llm:
            return RuleBasedBloomClassifier().predict(question)

        prompt = f"""You are an expert Bloom's Taxonomy evaluator for university examinations.
Classify the following exam question into exactly ONE of the six cognitive levels:
- Remember
- Understand
- Apply
- Analyze
- Evaluate
- Create

Question: "{question}"

Instructions:
Respond with ONLY the exact name of the cognitive level (one word). Do not output any explanation or punctuation.

Cognitive Level:"""

        try:
            resp = self.llm.generate([prompt])
            text = resp.generations[0][0].text.strip()
            # Normalize response to canonical levels
            for level in BLOOM_LEVELS:
                if level.lower() in text.lower():
                    return level
        except Exception as e:
            print(f"[ZeroShotLLM] Warning: Ollama call failed: {e}")

        # Fallback to rule-based on failure
        return RuleBasedBloomClassifier().predict(question)

    def predict_batch(self, questions: List[str]) -> List[str]:
        return [self.predict(q) for q in questions]


def get_bloom_classifier(model_type: str = "best") -> Any:
    """Retrieve an initialized Bloom's Taxonomy classifier instance.

    Args:
        model_type: 'best' (loads persisted model from disk), 'rule_based',
                    'tfidf', 'embedding', or 'llm'.

    Returns:
        Classifier instance implementing .predict(question).
    """
    if model_type == "rule_based":
        return RuleBasedBloomClassifier()
    elif model_type == "tfidf":
        return TFIDFBloomClassifier()
    elif model_type == "embedding":
        return EmbeddingBloomClassifier()
    elif model_type == "llm":
        return ZeroShotLLMBloomClassifier()

    # 'best' model loader
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            loaded_model = joblib.load(MODEL_SAVE_PATH)
            return loaded_model
        except Exception as e:
            print(f"[get_bloom_classifier] Notice: Could not load saved model ({e}). Using rule-based fallback.")

    return RuleBasedBloomClassifier()

