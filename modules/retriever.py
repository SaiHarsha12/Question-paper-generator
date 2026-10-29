"""RAG Retrieval Subsystem: Dense Semantic (FAISS) & Sparse (TF-IDF) Retrievers.

Course: Natural Language Processing for LLM (24AM7PCLLM)
Institution: B.M.S. College of Engineering (BMSCE)
Department: Artificial Intelligence & Machine Learning

This module implements the retrieval engine of the RAG pipeline:
1. Dense Semantic Retriever (KnowledgeBase):
   - Sentence-Transformers ('all-MiniLM-L6-v2') dense embeddings.
   - FAISS vector store with cosine similarity via L2-normalized IndexFlatIP.
   - Persistent index serialization (save/load).
2. Sparse Lexical Retriever (TFIDFRetriever):
   - Scikit-learn TF-IDF n-gram vectorizer with cosine similarity.
   - Identical API for rigorous comparative benchmarking.
3. Information Retrieval (IR) Evaluation Engine:
   - Precision@k, Recall@k, and Mean Reciprocal Rank (MRR) comparisons.
4. CLI support for ingestion and querying reference textbooks.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import glob
import json
import pickle
import argparse
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.preprocess import extract_text, clean_text, chunk_text


class KnowledgeBase:
    """Dense Semantic Knowledge Base using Sentence-Transformers and FAISS.

    Implements cosine similarity search over sentence-aware text chunks by
    L2-normalizing dense vectors and indexing them in FAISS IndexFlatIP.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the dense retriever with a SentenceTransformer encoder."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict[str, Any]] = []

    def build_from_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Build the FAISS vector index directly from pre-processed chunk dictionaries.

        Args:
            chunks: List of chunk dicts [{'chunk_id': str, 'text': str, 'source': str, 'page': int}].

        Returns:
            Number of chunks successfully indexed.
        """
        if not chunks:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunks = []
            return 0

        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        # Compute L2-normalized embeddings for inner product cosine equivalence
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        return len(self.chunks)

    def build(self, pdf_paths: List[str], chunk_size: int = 500, overlap: int = 100) -> int:
        """Extract, clean, chunk, and embed reference PDF documents into the index.

        Args:
            pdf_paths: List of file paths to PDF documents.
            chunk_size: Target character length per chunk.
            overlap: Character overlap between consecutive chunks.

        Returns:
            Total number of chunks indexed.
        """
        all_chunks: List[Dict[str, Any]] = []

        for path in pdf_paths:
            if not os.path.exists(path):
                print(f"[KnowledgeBase] Warning: PDF path not found: {path}")
                continue

            doc_name = os.path.basename(path)
            pages_data = extract_text(path, return_pages=True)

            if isinstance(pages_data, list):
                for p in pages_data:
                    raw_text = p.get("text", "")
                    cleaned = clean_text(raw_text)
                    if cleaned:
                        page_chunks = chunk_text(
                            cleaned,
                            chunk_size=chunk_size,
                            overlap=overlap,
                            source=doc_name,
                            page=p.get("page")
                        )
                        all_chunks.extend(page_chunks)
            else:
                cleaned = clean_text(str(pages_data))
                if cleaned:
                    doc_chunks = chunk_text(
                        cleaned,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        source=doc_name
                    )
                    all_chunks.extend(doc_chunks)

        return self.build_from_chunks(all_chunks)

    def save(self, dir_path: str = "data/index_store") -> None:
        """Persist the FAISS index and chunk metadata to disk."""
        os.makedirs(dir_path, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(dir_path, "index.faiss"))

        with open(os.path.join(dir_path, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

        config = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "total_chunks": len(self.chunks)
        }
        with open(os.path.join(dir_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def load(self, dir_path: str = "data/index_store") -> None:
        """Load a persisted FAISS index and chunk metadata from disk."""
        index_file = os.path.join(dir_path, "index.faiss")
        chunks_file = os.path.join(dir_path, "chunks.json")

        if not os.path.exists(index_file) or not os.path.exists(chunks_file):
            raise FileNotFoundError(f"KnowledgeBase index files not found in {dir_path}")

        self.index = faiss.read_index(index_file)
        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def retrieve(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[str, float, str, Optional[int]]]:
        """Retrieve top-k semantically relevant chunks for a given query.

        Args:
            query: Question or subtopic string.
            k: Number of relevant chunks to return.

        Returns:
            List of tuples: [(chunk_text, cosine_similarity_score, source, page)].
        """
        if not self.index or self.index.ntotal == 0 or not self.chunks:
            return []

        clean_q = clean_text(query)
        q_vec = self.model.encode(
            [clean_q],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        actual_k = min(k, self.index.ntotal)
        scores, indices = self.index.search(q_vec, actual_k)

        results: List[Tuple[str, float, str, Optional[int]]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append((
                chunk["text"],
                float(score),
                chunk.get("source", "unknown"),
                chunk.get("page")
            ))

        return results


class TFIDFRetriever:
    """Sparse Lexical Information Retrieval engine using Scikit-Learn TF-IDF.

    Provides identical retrieval interfaces as KnowledgeBase to enable side-by-side
    comparison of classical bag-of-words retrieval vs dense neural embeddings.
    """

    def __init__(self, ngram_range: Tuple[int, int] = (1, 2)):
        """Initialize TF-IDF vectorizer with unigram and bigram features."""
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=ngram_range,
            sublinear_tf=True
        )
        self.chunks: List[Dict[str, Any]] = []
        self.tfidf_matrix = None

    def build_from_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Build TF-IDF inverted index from pre-processed chunk dictionaries."""
        if not chunks:
            self.chunks = []
            self.tfidf_matrix = None
            return 0

        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        return len(self.chunks)

    def fit(self, chunks: List[Dict[str, Any]]) -> int:
        """Convenience alias for build_from_chunks."""
        return self.build_from_chunks(chunks)

    def build(self, pdf_paths: List[str], chunk_size: int = 500, overlap: int = 100) -> int:
        """Extract and index text chunks from PDF documents."""
        all_chunks: List[Dict[str, Any]] = []
        for path in pdf_paths:
            if not os.path.exists(path):
                continue
            doc_name = os.path.basename(path)
            pages = extract_text(path, return_pages=True)
            if isinstance(pages, list):
                for p in pages:
                    cleaned = clean_text(p.get("text", ""))
                    if cleaned:
                        all_chunks.extend(chunk_text(
                            cleaned, chunk_size=chunk_size, overlap=overlap, source=doc_name, page=p.get("page")
                        ))
            else:
                cleaned = clean_text(str(pages))
                if cleaned:
                    all_chunks.extend(chunk_text(
                        cleaned, chunk_size=chunk_size, overlap=overlap, source=doc_name
                    ))

        return self.build_from_chunks(all_chunks)

    def save(self, dir_path: str = "data/index_store") -> None:
        """Persist vectorizer, sparse matrix, and chunks."""
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "tfidf_model.pkl"), "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.tfidf_matrix}, f)
        with open(os.path.join(dir_path, "tfidf_chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

    def load(self, dir_path: str = "data/index_store") -> None:
        """Load vectorizer, sparse matrix, and chunks."""
        model_file = os.path.join(dir_path, "tfidf_model.pkl")
        chunks_file = os.path.join(dir_path, "tfidf_chunks.json")
        if not os.path.exists(model_file) or not os.path.exists(chunks_file):
            raise FileNotFoundError(f"TF-IDF model files not found in {dir_path}")

        with open(model_file, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.tfidf_matrix = data["matrix"]

        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def retrieve(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[str, float, str, Optional[int]]]:
        """Retrieve top-k chunks using cosine similarity over sparse TF-IDF vectors."""
        if self.tfidf_matrix is None or not self.chunks:
            return []

        clean_q = clean_text(query)
        q_vec = self.vectorizer.transform([clean_q])
        cos_sim = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(cos_sim)[::-1][:k]

        results: List[Tuple[str, float, str, Optional[int]]] = []
        for idx in top_indices:
            score = float(cos_sim[idx])
            if score <= 0.0:
                continue
            chunk = self.chunks[idx]
            results.append((
                chunk["text"],
                score,
                chunk.get("source", "unknown"),
                chunk.get("page")
            ))

        return results


def compare_retrievers(
    queries: List[str],
    ground_truth: List[List[str]],
    dense_retriever: Optional[KnowledgeBase] = None,
    tfidf_retriever: Optional[TFIDFRetriever] = None,
    k: int = 4
) -> Dict[str, Any]:
    """Evaluate and compare Dense Semantic vs Sparse Lexical Retrieval performance.

    Computes:
    - Precision@k: fraction of retrieved chunks that are relevant.
    - Recall@k: fraction of relevant target concepts successfully retrieved.
    - Mean Reciprocal Rank (MRR): reciprocal rank of the first relevant retrieved chunk.

    Args:
        queries: List of test queries.
        ground_truth: List of relevant keywords/concepts expected for each query.
        dense_retriever: Optional KnowledgeBase instance (defaults to loaded data/index_store).
        tfidf_retriever: Optional TFIDFRetriever instance (defaults to loaded data/index_store).
        k: Top-k cutoff.

    Returns:
        Structured evaluation dictionary containing per-query metrics and macro-averages.
    """
    assert len(queries) == len(ground_truth), "Queries and ground_truth lists must have identical lengths"

    if dense_retriever is None:
        dense_retriever = KnowledgeBase()
        if os.path.exists("data/index_store/index.faiss"):
            dense_retriever.load("data/index_store")

    if tfidf_retriever is None:
        tfidf_retriever = TFIDFRetriever()
        if os.path.exists("data/index_store/tfidf_model.pkl"):
            tfidf_retriever.load("data/index_store")

    dense_precisions: List[float] = []
    dense_recalls: List[float] = []
    dense_rr: List[float] = []

    tfidf_precisions: List[float] = []
    tfidf_recalls: List[float] = []
    tfidf_rr: List[float] = []

    per_query_results: List[Dict[str, Any]] = []

    for query, target_keywords in zip(queries, ground_truth):
        # 1. Evaluate Dense
        dense_results = dense_retriever.retrieve(query, k=k)
        dense_p, dense_r, dense_recip = _score_retrieval(dense_results, target_keywords, k)
        dense_precisions.append(dense_p)
        dense_recalls.append(dense_r)
        dense_rr.append(dense_recip)

        # 2. Evaluate TF-IDF
        tfidf_results = tfidf_retriever.retrieve(query, k=k)
        tfidf_p, tfidf_r, tfidf_recip = _score_retrieval(tfidf_results, target_keywords, k)
        tfidf_precisions.append(tfidf_p)
        tfidf_recalls.append(tfidf_r)
        tfidf_rr.append(tfidf_recip)

        per_query_results.append({
            "query": query,
            "ground_truth_targets": target_keywords,
            "dense": {
                "precision_at_k": dense_p,
                "recall_at_k": dense_r,
                "reciprocal_rank": dense_recip,
                "top_retrieved": [t[0][:60] + "..." for t in dense_results[:2]]
            },
            "tfidf": {
                "precision_at_k": tfidf_p,
                "recall_at_k": tfidf_r,
                "reciprocal_rank": tfidf_recip,
                "top_retrieved": [t[0][:60] + "..." for t in tfidf_results[:2]]
            }
        })

    def _mean(vals: List[float]) -> float:
        return round(float(np.mean(vals)), 4) if vals else 0.0

    out = {
        "k": k,
        "total_queries_evaluated": len(queries),
        "dense_embeddings_all_MiniLM_L6_v2": {
            "mean_precision_at_k": _mean(dense_precisions),
            "mean_recall_at_k": _mean(dense_recalls),
            "mean_reciprocal_rank_MRR": _mean(dense_rr)
        },
        "sparse_lexical_TFIDF": {
            "mean_precision_at_k": _mean(tfidf_precisions),
            "mean_recall_at_k": _mean(tfidf_recalls),
            "mean_reciprocal_rank_MRR": _mean(tfidf_rr)
        },
        "query_level_observations": per_query_results
    }
    out["dense"] = out["dense_embeddings_all_MiniLM_L6_v2"]
    out["tfidf"] = out["sparse_lexical_TFIDF"]
    return out


def _score_retrieval(
    retrieved_items: List[Tuple[str, float, str, Optional[int]]],
    target_keywords: List[str],
    k: int
) -> Tuple[float, float, float]:
    """Helper calculating Precision@k, Recall@k, and Reciprocal Rank."""
    if not retrieved_items or not target_keywords:
        return 0.0, 0.0, 0.0

    relevant_retrieved = 0
    hit_targets = set()
    first_rank = 0

    for rank, (text, _, _, _) in enumerate(retrieved_items[:k], start=1):
        text_lower = text.lower()
        matched = False
        for kw in target_keywords:
            if kw.lower() in text_lower:
                matched = True
                hit_targets.add(kw.lower())

        if matched:
            relevant_retrieved += 1
            if first_rank == 0:
                first_rank = rank

    precision = round(relevant_retrieved / k, 4)
    recall = round(len(hit_targets) / max(len(target_keywords), 1), 4)
    mrr = round(1.0 / first_rank, 4) if first_rank > 0 else 0.0

    return precision, recall, mrr


# ----------------------------------------------------------------------
# CLI Interface
# ----------------------------------------------------------------------
def main() -> None:
    """CLI driver for building and testing the KnowledgeBase index."""
    parser = argparse.ArgumentParser(
        description="Build or query KnowledgeBase vector index for RAG Question Paper Generation."
    )
    parser.add_argument(
        "--build",
        nargs="+",
        help="One or more PDF paths or glob patterns (e.g., data/reference/*.pdf)"
    )
    parser.add_argument(
        "--save-dir",
        default="data/index_store",
        help="Directory to save/load index files (default: data/index_store)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to search against the knowledge base"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of retrieved chunks (default: 4)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run benchmark comparison between Dense and TF-IDF retrievers"
    )

    args = parser.parse_args()

    # Collect files if --build specified
    if args.build:
        resolved_files: List[str] = []
        for pattern in args.build:
            matched = glob.glob(pattern)
            if matched:
                resolved_files.extend(matched)
            elif os.path.exists(pattern):
                resolved_files.append(pattern)

        if not resolved_files:
            print(f"Error: No PDF files matched arguments: {args.build}")
            return

        print(f"Indexing {len(resolved_files)} documents into KnowledgeBase...")
        kb = KnowledgeBase()
        count = kb.build(resolved_files)
        kb.save(args.save_dir)
        print(f"Successfully indexed {count} chunks and saved FAISS index to {args.save_dir}")

        tfidf = TFIDFRetriever()
        tfidf.build_from_chunks(kb.chunks)
        tfidf.save(args.save_dir)
        print(f"Successfully built and saved TF-IDF model to {args.save_dir}")

    # Query if requested
    if args.query:
        kb = KnowledgeBase()
        kb.load(args.save_dir)
        results = kb.retrieve(args.query, k=args.k)
        print(f"\nTop-{len(results)} Dense Semantic Results for '{args.query}':\n" + "=" * 60)
        for i, (txt, score, src, page) in enumerate(results, start=1):
            p_str = f"Page {page}" if page else "Page N/A"
            print(f"[{i}] Score: {score:.4f} | Source: {src} ({p_str})\n{txt}\n" + "-" * 60)

    # Benchmark comparison if requested
    if args.compare:
        kb = KnowledgeBase()
        kb.load(args.save_dir)
        tfidf = TFIDFRetriever()
        tfidf.load(args.save_dir)

        test_queries = [
            "routing protocols distance vector link state",
            "cryptography public key RSA encryption",
            "transport layer TCP flow control congestion"
        ]
        ground_truth = [
            ["routing", "distance vector", "link state"],
            ["cryptography", "key", "encryption", "cipher"],
            ["tcp", "transport", "congestion", "protocol"]
        ]
        comparison = compare_retrievers(test_queries, ground_truth, kb, tfidf, k=args.k)
        print("\n=== RETRIEVER COMPARISON BENCHMARK ===")
        print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

