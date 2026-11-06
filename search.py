"""
Smart Notes Assistant - Search Module
Provides Keyword-based search and TF-IDF Cosine Similarity search,
along with Hit Rate @ 3 and Mean Reciprocal Rank (MRR) evaluation.
"""

import math
import re
from typing import List, Dict, Any, Optional
from classifier import preprocess_text, STOP_WORDS


def _tokenize_words(text: str) -> List[str]:
    cleaned = preprocess_text(text)
    return [w for w in cleaned.split() if w]


class NoteSearchEngine:
    """
    Search engine supporting both exact/fuzzy Keyword matching and
    vectorized TF-IDF Cosine Similarity search.
    """

    def __init__(self, notes: Optional[List[Dict[str, Any]]] = None):
        self.notes: List[Dict[str, Any]] = notes or []
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[int, float]] = []
        if self.notes:
            self.index_notes(self.notes)

    def index_notes(self, notes: List[Dict[str, Any]]):
        """Build TF-IDF inverted index over all notes."""
        self.notes = list(notes)
        if not self.notes:
            self.vocabulary = {}
            self.idf = {}
            self.doc_vectors = []
            return

        doc_tokens = [_tokenize_words(n.get("text", "")) for n in self.notes]
        n_docs = len(doc_tokens)

        # Document frequency
        df: Dict[str, int] = {}
        for tokens in doc_tokens:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        self.vocabulary = {term: idx for idx, term in enumerate(sorted(df.keys()))}
        # Smooth IDF
        self.idf = {
            term: math.log((1.0 + n_docs) / (1.0 + count)) + 1.0
            for term, count in df.items()
        }

        # Vectorize each note
        self.doc_vectors = []
        for tokens in doc_tokens:
            vec = self._vectorize_tokens(tokens)
            self.doc_vectors.append(vec)

    def _vectorize_tokens(self, tokens: List[str]) -> Dict[int, float]:
        tf: Dict[str, int] = {}
        for t in tokens:
            if t in self.vocabulary:
                tf[t] = tf.get(t, 0) + 1

        vec: Dict[int, float] = {}
        norm_sq = 0.0
        for t, count in tf.items():
            idx = self.vocabulary[t]
            val = (1.0 + math.log(count)) * self.idf.get(t, 1.0)
            vec[idx] = val
            norm_sq += val * val

        norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
        for idx in vec:
            vec[idx] /= norm
        return vec

    def keyword_search(
        self,
        query: str,
        category_filter: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Keyword-based search. Scores notes based on exact and partial token overlap,
        sub-string matching, and title keyword frequency.
        """
        if not query.strip():
            return []

        query_tokens = _tokenize_words(query)
        if not query_tokens:
            # Fall back to raw terms
            query_tokens = [w.lower() for w in re.findall(r"\w+", query) if w]

        results = []
        for note in self.notes:
            if category_filter and category_filter != "All" and note.get("category") != category_filter:
                continue

            text = note.get("text", "")
            lower_text = text.lower()
            doc_tokens = set(_tokenize_words(text))

            score = 0.0
            matched_terms = []

            # Exact phrase bonus
            if query.lower().strip() in lower_text:
                score += 2.0

            for q_term in query_tokens:
                if q_term in doc_tokens:
                    score += 1.0
                    matched_terms.append(q_term)
                elif any(q_term in doc_t or doc_t in q_term for doc_t in doc_tokens):
                    score += 0.5
                    matched_terms.append(q_term)

            if score > 0:
                # Normalize score to 0.0 - 1.0 range
                norm_score = min(round(score / (len(query_tokens) + 1.0), 4), 1.0)
                res = dict(note)
                res["relevance_score"] = norm_score
                res["search_method"] = "keyword"
                res["matched_terms"] = list(set(matched_terms))
                results.append(res)

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]

    def similarity_search(
        self,
        query: str,
        category_filter: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        TF-IDF Cosine Similarity search. Measures geometric angular closeness
        in high-dimensional vector space. Excels at natural language matching.
        """
        if not query.strip() or not self.notes or not self.vocabulary:
            return []

        q_tokens = _tokenize_words(query)
        if not q_tokens:
            return []

        q_vec = self._vectorize_tokens(q_tokens)
        if not q_vec:
            # Query has no overlapping vocabulary words; fallback to partial keyword matching
            return self.keyword_search(query, category_filter, top_k)

        results = []
        for idx, note in enumerate(self.notes):
            if category_filter and category_filter != "All" and note.get("category") != category_filter:
                continue

            doc_vec = self.doc_vectors[idx] if idx < len(self.doc_vectors) else {}
            # Cosine similarity (dot product of L2-normalized vectors)
            sim = 0.0
            for feat_idx, q_val in q_vec.items():
                if feat_idx in doc_vec:
                    sim += q_val * doc_vec[feat_idx]

            if sim > 0.01:
                res = dict(note)
                res["relevance_score"] = round(float(sim), 4)
                res["search_method"] = "tfidf_similarity"
                results.append(res)

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]

    def search(
        self,
        query: str,
        method: str = "tfidf_similarity",
        category_filter: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Unified search interface with method selection."""
        if method == "keyword":
            return self.keyword_search(query, category_filter, top_k)
        return self.similarity_search(query, category_filter, top_k)

    def evaluate_retrieval(
        self,
        benchmark_queries: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate and compare Keyword vs TF-IDF search using:
        - Hit Rate @ 1
        - Hit Rate @ 3
        - Mean Reciprocal Rank (MRR)
        """
        if benchmark_queries is None:
            # Standard realistic benchmark queries
            benchmark_queries = [
                {"query": "electricity payment", "expected_category": "Finance", "keyword": "bill"},
                {"query": "machine learning exam", "expected_category": "Education", "keyword": "python"},
                {"query": "family call", "expected_category": "Personal", "keyword": "grandma"},
                {"query": "flight to Delhi", "expected_category": "Travel", "keyword": "delhi"},
                {"query": "buy grocery items", "expected_category": "Shopping", "keyword": "groceries"},
                {"query": "dentist appointment", "expected_category": "Health", "keyword": "dentist"},
                {"query": "birthday celebration party", "expected_category": "Events", "keyword": "birthday"},
                {"query": "git pull request review", "expected_category": "Work", "keyword": "request"},
            ]

        def eval_method(method_name: str) -> Dict[str, float]:
            hits_at_1 = 0
            hits_at_3 = 0
            reciprocal_ranks = []

            for item in benchmark_queries:
                q = item["query"]
                exp_cat = item.get("expected_category")
                kw = item.get("keyword", "").lower()

                if method_name == "keyword":
                    res = self.keyword_search(q, top_k=5)
                else:
                    res = self.similarity_search(q, top_k=5)

                # Check if target is retrieved
                rank = None
                for r_idx, hit in enumerate(res):
                    text_lower = hit.get("text", "").lower()
                    cat = hit.get("category")
                    # Match if expected keyword in text or expected category matches
                    if kw in text_lower or cat == exp_cat:
                        rank = r_idx + 1
                        break

                if rank == 1:
                    hits_at_1 += 1
                if rank is not None and rank <= 3:
                    hits_at_3 += 1

                if rank is not None:
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)

            n = len(benchmark_queries) if benchmark_queries else 1
            return {
                "hit_rate_at_1": round(hits_at_1 / n, 4),
                "hit_rate_at_3": round(hits_at_3 / n, 4),
                "mrr": round(sum(reciprocal_ranks) / n, 4),
                "total_queries": n,
            }

        kw_metrics = eval_method("keyword")
        tfidf_metrics = eval_method("tfidf_similarity")

        return {
            "keyword_search": kw_metrics,
            "tfidf_similarity": tfidf_metrics,
            "comparison_summary": (
                f"TF-IDF Cosine Similarity achieved {tfidf_metrics['hit_rate_at_3']*100:.1f}% Hit Rate @ 3 "
                f"vs {kw_metrics['hit_rate_at_3']*100:.1f}% for Keyword matching on natural language queries."
            ),
        }
