"""
Smart Notes Assistant - ML Classifier Module
Handles text preprocessing, TF-IDF vectorization, Logistic Regression classification,
uncertainty detection, and model evaluation metrics.
"""

import os
import re
import math
import pickle
from typing import Dict, List, Tuple, Any, Optional

CATEGORIES = [
    "Education",
    "Work",
    "Personal",
    "Finance",
    "Travel",
    "Shopping",
    "Health",
    "Events",
]

DEFAULT_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_MARGIN_THRESHOLD = 0.12

# Standard English stop words
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me",
    "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


def preprocess_text(text: str) -> str:
    """
    Preprocess raw text: lowercase, remove special characters/punctuation,
    and normalize whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Replace punctuation and special characters with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Remove excess whitespace
    tokens = [token for token in text.split() if token and token not in STOP_WORDS]
    return " ".join(tokens)


class NoteClassifier:
    """
    Machine Learning note classifier using TF-IDF feature extraction and
    Logistic Regression. Supports uncertainty detection and model persistence.
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    ):
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.categories = list(CATEGORIES)
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.weights: Dict[str, List[float]] = {}  # category -> weight vector for vocab
        self.biases: Dict[str, float] = {}         # category -> bias
        self.is_trained: bool = False
        self.n_samples_trained: int = 0
        self.sklearn_pipeline = None

    def _tokenize(self, text: str) -> List[str]:
        cleaned = preprocess_text(text)
        words = cleaned.split()
        # Unigrams + Bigrams for better context capture
        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
        return tokens

    def _compute_tfidf_vector(self, text: str) -> Dict[int, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        
        # Term frequencies
        tf: Dict[str, int] = {}
        for token in tokens:
            if token in self.vocabulary:
                tf[token] = tf.get(token, 0) + 1
        
        # TF-IDF with sublinear scaling (1 + log(tf))
        vec: Dict[int, float] = {}
        norm_sq = 0.0
        for token, count in tf.items():
            idx = self.vocabulary[token]
            val = (1.0 + math.log(count)) * self.idf.get(token, 1.0)
            vec[idx] = val
            norm_sq += val * val
        
        # L2 normalize
        norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
        for idx in vec:
            vec[idx] /= norm
            
        return vec

    def fit(self, texts: List[str], labels: List[str], max_iter: int = 200, lr: float = 0.5):
        """
        Train the TF-IDF feature representation and Logistic Regression classifier.
        Uses Scikit-learn if available; otherwise uses optimized pure-Python logistic regression.
        """
        if len(texts) != len(labels) or not texts:
            raise ValueError("Texts and labels must be non-empty and of matching length.")

        # Try sklearn first if installed
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline

            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
            ])
            pipeline.fit(texts, labels)
            self.sklearn_pipeline = pipeline
            self.categories = list(pipeline.classes_)
            self.is_trained = True
            self.n_samples_trained = len(texts)
            return self
        except ImportError:
            # Seamless pure-Python implementation
            pass

        # 1. Build vocabulary & compute document frequencies (DF)
        doc_count = len(texts)
        df: Dict[str, int] = {}
        tokenized_docs = [self._tokenize(t) for t in texts]

        for doc_tokens in tokenized_docs:
            seen = set(doc_tokens)
            for token in seen:
                df[token] = df.get(token, 0) + 1

        # Filter vocabulary (terms appearing at least once)
        vocab_list = sorted([term for term, count in df.items() if count >= 1])
        self.vocabulary = {term: idx for idx, term in enumerate(vocab_list)}
        
        # Compute smooth IDF: log((1 + N) / (1 + df)) + 1
        self.idf = {
            term: math.log((1.0 + doc_count) / (1.0 + df[term])) + 1.0
            for term in self.vocabulary
        }

        # 2. Build feature matrix (sparse TF-IDF vectors)
        X_vecs = [self._compute_tfidf_vector(t) for t in texts]

        # 3. Train multi-class Logistic Regression with One-vs-Rest / Softmax gradient descent
        unique_categories = sorted(list(set(labels)))
        self.categories = unique_categories
        vocab_size = len(self.vocabulary)

        self.weights = {cat: [0.0] * vocab_size for cat in self.categories}
        self.biases = {cat: 0.0 for cat in self.categories}

        # Gradient descent optimization
        for _ in range(max_iter):
            for i, x_vec in enumerate(X_vecs):
                target_cat = labels[i]
                # Compute raw scores (logits)
                logits: Dict[str, float] = {}
                for cat in self.categories:
                    w = self.weights[cat]
                    score = self.biases[cat]
                    for feat_idx, val in x_vec.items():
                        score += w[feat_idx] * val
                    logits[cat] = score

                # Softmax
                max_logit = max(logits.values())
                exp_scores = {c: math.exp(logits[c] - max_logit) for c in self.categories}
                sum_exp = sum(exp_scores.values())
                probs = {c: exp_scores[c] / sum_exp for c in self.categories}

                # Update weights with gradient step and L2 regularization
                for cat in self.categories:
                    y = 1.0 if cat == target_cat else 0.0
                    grad = probs[cat] - y
                    # Update bias
                    self.biases[cat] -= lr * grad
                    # Update weights
                    w = self.weights[cat]
                    for feat_idx, val in x_vec.items():
                        # L2 penalty 0.001 * w
                        w[feat_idx] -= lr * (grad * val + 0.001 * w[feat_idx])

        self.is_trained = True
        self.n_samples_trained = len(texts)
        return self

    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Predict probability distribution over categories for a single note.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        if self.sklearn_pipeline is not None:
            probas = self.sklearn_pipeline.predict_proba([text])[0]
            classes = list(self.sklearn_pipeline.classes_)
            return {cls: float(p) for cls, p in zip(classes, probas)}

        x_vec = self._compute_tfidf_vector(text)
        logits: Dict[str, float] = {}
        for cat in self.categories:
            score = self.biases.get(cat, 0.0)
            w = self.weights.get(cat, [])
            for feat_idx, val in x_vec.items():
                if feat_idx < len(w):
                    score += w[feat_idx] * val
            logits[cat] = score

        max_logit = max(logits.values()) if logits else 0.0
        exp_scores = {c: math.exp(logits[c] - max_logit) for c in self.categories}
        sum_exp = sum(exp_scores.values()) if sum(exp_scores.values()) > 0 else 1.0
        return {c: exp_scores[c] / sum_exp for c in self.categories}

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict the most likely category for a note.
        Includes confidence score, sorted probabilities, and uncertainty detection.
        """
        probas = self.predict_proba(text)
        sorted_cats = sorted(probas.items(), key=lambda item: item[1], reverse=True)
        top_cat, top_conf = sorted_cats[0]
        second_conf = sorted_cats[1][1] if len(sorted_cats) > 1 else 0.0
        margin = top_conf - second_conf

        # Uncertainty check:
        # 1. Top confidence is below threshold
        # 2. Or margin between #1 and #2 is too tight
        # 3. Or text is very short / ambiguous
        cleaned = preprocess_text(text)
        is_ambiguous = (
            top_conf < self.confidence_threshold
            or margin < self.margin_threshold
            or len(cleaned.split()) == 0
        )

        uncertainty_message = ""
        if is_ambiguous:
            uncertainty_message = (
                "Not sure about this note's category. Please select a category manually."
            )

        return {
            "category": top_cat,
            "confidence": round(top_conf, 4),
            "margin": round(margin, 4),
            "is_uncertain": is_ambiguous,
            "uncertainty_message": uncertainty_message,
            "all_probabilities": {k: round(v, 4) for k, v in sorted_cats},
        }

    def evaluate(self, test_texts: List[str], test_labels: List[str]) -> Dict[str, Any]:
        """
        Evaluate model using accuracy, precision, recall, macro-F1, and confusion matrix.
        """
        if not test_texts or len(test_texts) != len(test_labels):
            raise ValueError("Test texts and labels must be valid and of matching length.")

        preds = [self.predict(t)["category"] for t in test_texts]

        # Use scikit-learn metrics if available
        try:
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
            acc = float(accuracy_score(test_labels, preds))
            prec, rec, f1, _ = precision_recall_fscore_support(
                test_labels, preds, average="macro", zero_division=0
            )
            cm = confusion_matrix(test_labels, preds, labels=self.categories).tolist()
            
            per_class_prec, per_class_rec, per_class_f1, support = precision_recall_fscore_support(
                test_labels, preds, labels=self.categories, zero_division=0
            )
            per_class = {
                cat: {
                    "precision": round(float(p), 4),
                    "recall": round(float(r), 4),
                    "f1_score": round(float(f), 4),
                    "support": int(s),
                }
                for cat, p, r, f, s in zip(
                    self.categories, per_class_prec, per_class_rec, per_class_f1, support
                )
            }
            return {
                "accuracy": round(acc, 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "macro_f1": round(float(f1), 4),
                "confusion_matrix": cm,
                "categories": self.categories,
                "per_class_metrics": per_class,
                "total_samples": len(test_texts),
            }
        except ImportError:
            pass

        # Standalone computation of evaluation metrics
        total = len(test_labels)
        correct = sum(1 for y_true, y_pred in zip(test_labels, preds) if y_true == y_pred)
        accuracy = correct / total if total > 0 else 0.0

        # Confusion Matrix [true_idx][pred_idx]
        cat_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        k = len(self.categories)
        cm = [[0 for _ in range(k)] for _ in range(k)]

        for y_true, y_pred in zip(test_labels, preds):
            if y_true in cat_to_idx and y_pred in cat_to_idx:
                cm[cat_to_idx[y_true]][cat_to_idx[y_pred]] += 1

        # Per-class precision, recall, f1
        per_class = {}
        precisions, recalls, f1s = [], [], []

        for i, cat in enumerate(self.categories):
            tp = cm[i][i]
            fp = sum(cm[r][i] for r in range(k)) - tp
            fn = sum(cm[i][c] for c in range(k)) - tp
            support = sum(cm[i][c] for c in range(k))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

            per_class[cat] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "support": support,
            }

        macro_prec = sum(precisions) / k if k > 0 else 0.0
        macro_rec = sum(recalls) / k if k > 0 else 0.0
        macro_f1 = sum(f1s) / k if k > 0 else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(macro_prec, 4),
            "recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "confusion_matrix": cm,
            "categories": self.categories,
            "per_class_metrics": per_class,
            "total_samples": total,
        }

    def save(self, filepath: str):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "NoteClassifier":
        """Load trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        with open(filepath, "rb") as f:
            instance = pickle.load(f)
        return instance
