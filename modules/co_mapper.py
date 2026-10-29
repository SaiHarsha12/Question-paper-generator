"""Course Outcome (CO) and Programme Outcome (PO) Semantic Mapping Subsystem.

Course: Natural Language Processing for LLM (24AM7PCLLM)
Institution: B.M.S. College of Engineering (BMSCE)
Department: Artificial Intelligence & Machine Learning

This module provides automated semantic mapping from exam questions to
Course Outcomes (COs) and Programme Outcomes (POs / PSOs):
1. Embeds question text and candidate CO descriptions using Sentence-Transformers (all-MiniLM-L6-v2).
2. Computes normalized cosine similarity between question vectors and CO vectors.
3. Optionally blends semantic similarity with Bloom-level pedagogical priors:
   - Remember / Understand / Apply -> CO1
   - Analyze -> CO2
   - Evaluate -> CO3
   - Create -> CO4
4. Maps the selected CO to its corresponding PO/PSO vector via the 24AM7PCLLM accreditation matrix.
5. Loads configuration dynamically from data/course_config.json with fallback defaults.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Default Course Outcomes and Accreditation Matrix for 24AM7PCLLM (BMSCE AIML)
DEFAULT_CONFIG: Dict[str, Any] = {
    "course_code": "24AM7PCLLM",
    "course_name": "Natural Language Processing for LLM",
    "semester": "VII",
    "branch": "Artificial Intelligence and Machine Learning",
    "cos": {
        "CO1": "Apply the various concepts and techniques on the input in the model to process and interpret the data.",
        "CO2": "Analyze the natural language problems using appropriate language modeling and computational approaches to identify linguistic patterns and contextual relationships.",
        "CO3": "Evaluate LLM adaptation and enhancement techniques to improve language understanding and generation.",
        "CO4": "Design, develop and demonstrate LLM-based solutions for real-world applications in teams."
    },
    "co_po_matrix": {
        "CO1": {"PO1": 3, "PSO3": 3},
        "CO2": {"PO2": 3, "PSO3": 3},
        "CO3": {"PO3": 2, "PO4": 3, "PSO3": 3},
        "CO4": {"PO3": 3, "PO4": 3, "PO5": 3, "PO8": 2, "PO9": 2, "PSO3": 3}
    },
    "bloom_co_prior": {
        "remember": "CO1",
        "understand": "CO1",
        "apply": "CO1",
        "analyze": "CO2",
        "evaluate": "CO3",
        "create": "CO4"
    }
}

# Module-level model cache to avoid expensive reload of sentence-transformers
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Retrieve or cache the SentenceTransformer embedding model."""
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def load_course_config(config_path: str = "data/course_config.json") -> Dict[str, Any]:
    """Load course configuration from JSON file, falling back to defaults if missing.
    
    Args:
        config_path: Filepath to course_config.json.
        
    Returns:
        Dictionary containing course_code, course_name, semester, branch, cos, co_po_matrix, bloom_co_prior.
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure all required top-level keys exist with defaults
                merged = dict(DEFAULT_CONFIG)
                merged.update(data)
                return merged
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_course_config(config: Dict[str, Any], config_path: str = "data/course_config.json") -> None:
    """Save updated course configuration to disk.
    
    Args:
        config: Configuration dictionary.
        config_path: Target JSON filepath.
    """
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


class COMapper:
    """Semantic Mapper matching exam questions to Course Outcomes and Accreditation POs."""

    def __init__(
        self,
        config_path: str = "data/course_config.json",
        cos: Optional[Dict[str, str]] = None,
        co_po_matrix: Optional[Dict[str, Dict[str, int]]] = None,
        bloom_prior_map: Optional[Dict[str, str]] = None,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """Initialize COMapper with configuration and pre-computed embeddings.
        
        Args:
            config_path: Path to course_config.json.
            cos: Optional explicit dictionary of CO keys to descriptions.
            co_po_matrix: Optional explicit mapping of CO to PO/PSO scores.
            bloom_prior_map: Optional mapping of bloom level to default CO.
            model_name: Sentence-Transformers model name.
        """
        cfg = load_course_config(config_path)
        self.cos: Dict[str, str] = cos if cos is not None else cfg.get("cos", DEFAULT_CONFIG["cos"])
        self.co_po_matrix: Dict[str, Dict[str, int]] = (
            co_po_matrix if co_po_matrix is not None else cfg.get("co_po_matrix", DEFAULT_CONFIG["co_po_matrix"])
        )
        self.bloom_prior_map: Dict[str, str] = (
            bloom_prior_map if bloom_prior_map is not None else cfg.get("bloom_co_prior", DEFAULT_CONFIG["bloom_co_prior"])
        )
        self.model_name = model_name
        self.model = get_embedding_model(model_name)

        # Precompute normalized embeddings for active Course Outcomes
        self.co_keys: List[str] = []
        self.co_embeddings: Optional[np.ndarray] = None
        self._embed_cos()

    def _embed_cos(self) -> None:
        """Encode and L2-normalize all active CO descriptions."""
        if not self.cos:
            self.co_keys = []
            self.co_embeddings = None
            return

        self.co_keys = list(self.cos.keys())
        descriptions = [str(self.cos[k]) for k in self.co_keys]
        embs = self.model.encode(descriptions, convert_to_numpy=True, normalize_embeddings=True)
        self.co_embeddings = np.array(embs, dtype=np.float32)

    def set_course_outcomes(self, cos: Dict[str, str]) -> None:
        """Update active Course Outcomes and re-index embeddings.
        
        Args:
            cos: Dict of CO key (e.g. 'CO1') to outcome text.
        """
        self.cos = cos
        self._embed_cos()

    def get_pos_for_co(self, co: str, include_pso: bool = False) -> List[str]:
        """Lookup mapped PO/PSO identifiers for a given CO.
        
        Args:
            co: Course Outcome identifier (e.g., 'CO3').
            include_pso: Whether to include PSO keys (e.g., 'PSO3') alongside PO keys.
            
        Returns:
            List of outcome strings (e.g., ['PO3', 'PO4'] or ['PO3', 'PO4', 'PSO3']).
        """
        mapping = self.co_po_matrix.get(co, {})
        if not mapping:
            # Fallback if unfamiliar CO
            return ["PO1"]

        if include_pso:
            return list(mapping.keys())

        pos = [k for k in mapping.keys() if k.startswith("PO")]
        return pos if pos else list(mapping.keys())

    def map_question(
        self,
        question: str,
        bloom_level: str = "",
        blend_prior: bool = True,
        alpha: float = 0.7,
        include_pso: bool = False
    ) -> Tuple[str, float, List[str]]:
        """Map an exam question to its closest Course Outcome and mapped PO list.
        
        Args:
            question: Cleaned exam question text.
            bloom_level: Pedagogical Bloom cognitive level (e.g. 'Apply', 'Analyze').
            blend_prior: Whether to blend semantic cosine similarity with the Bloom-level prior.
            alpha: Weight for semantic similarity vs. prior (1.0 = pure semantic similarity, 0.0 = pure prior).
            include_pso: Whether to include PSO keys in the returned list.
            
        Returns:
            Tuple of (co_identifier, cosine_similarity, po_list).
        """
        if not self.co_keys or self.co_embeddings is None:
            return "CO1", 0.0, ["PO1"]

        # Encode and normalize question embedding
        q_emb = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]

        # Cosine similarity via dot product of unit-normalized vectors
        sims = np.dot(self.co_embeddings, q_emb)

        # Blend with Bloom prior if applicable
        b_clean = str(bloom_level or "").strip().lower()
        prior_co = self.bloom_prior_map.get(b_clean)

        if blend_prior and prior_co:
            prior_weights = np.array(
                [1.0 if k.upper() == prior_co.upper() else 0.0 for k in self.co_keys],
                dtype=np.float32
            )
            # Convex combination of semantic cosine similarity and pedagogical prior
            scores = alpha * sims + (1.0 - alpha) * prior_weights
        else:
            scores = sims

        best_idx = int(np.argmax(scores))
        best_co = self.co_keys[best_idx]
        best_sim = float(sims[best_idx])
        po_list = self.get_pos_for_co(best_co, include_pso=include_pso)

        return best_co, round(best_sim, 4), po_list


# Module-level singleton
_DEFAULT_MAPPER: Optional[COMapper] = None


def get_co_mapper() -> COMapper:
    """Retrieve or initialize the default COMapper singleton."""
    global _DEFAULT_MAPPER
    if _DEFAULT_MAPPER is None:
        _DEFAULT_MAPPER = COMapper()
    return _DEFAULT_MAPPER


def map_question_to_co(
    question: str,
    bloom_level: str = "",
    cos: Optional[Dict[str, str]] = None,
    co_po_matrix: Optional[Dict[str, Dict[str, int]]] = None,
    blend_prior: bool = True,
    alpha: float = 0.7,
    include_pso: bool = False
) -> Tuple[str, float, List[str]]:
    """Map an exam question to the best matching Course Outcome and PO list.
    
    Args:
        question: Question text to evaluate.
        bloom_level: Bloom cognitive level (Remember, Understand, Apply, Analyze, Evaluate, Create).
        cos: Optional dictionary of COs. If None, loaded from course_config.json.
        co_po_matrix: Optional CO-PO matrix. If None, loaded from course_config.json.
        blend_prior: Whether to blend semantic similarity with Bloom prior.
        alpha: Weight for semantic cosine similarity (default 0.7).
        include_pso: Whether to include PSO keys (e.g. PSO3). Default False for standard PO column.
        
    Returns:
        Tuple of (co, similarity, po_list), for example ('CO3', 0.8124, ['PO3', 'PO4']).
    """
    if cos is not None or co_po_matrix is not None:
        mapper = COMapper(cos=cos, co_po_matrix=co_po_matrix)
        return mapper.map_question(
            question=question,
            bloom_level=bloom_level,
            blend_prior=blend_prior,
            alpha=alpha,
            include_pso=include_pso
        )

    mapper = get_co_mapper()
    return mapper.map_question(
        question=question,
        bloom_level=bloom_level,
        blend_prior=blend_prior,
        alpha=alpha,
        include_pso=include_pso
    )

