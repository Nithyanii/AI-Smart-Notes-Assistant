"""
Smart Notes Assistant - Model Training & Evaluation Pipeline
Loads dataset, splits train/test sets, fits TF-IDF & Logistic Regression,
computes evaluation metrics (Accuracy, Precision, Recall, Macro-F1, Confusion Matrix),
and saves the trained model to data/model.pkl.
"""

import os
import csv
import random
from typing import List, Tuple
from classifier import NoteClassifier, CATEGORIES

DATASET_PATH = "data/training_notes.csv"
MODEL_SAVE_PATH = "data/model.pkl"


def load_dataset(csv_path: str = DATASET_PATH) -> Tuple[List[str], List[str]]:
    """Load notes and ground-truth categories from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    texts, labels = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("text", "").strip()
            c = row.get("category", "").strip()
            if t and c:
                texts.append(t)
                labels.append(c)

    return texts, labels


def stratified_split(
    texts: List[str], labels: List[str], test_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Stratified train/test split to ensure all categories are represented."""
    random.seed(seed)
    by_category = {}
    for t, l in zip(texts, labels):
        by_category.setdefault(l, []).append(t)

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for cat, items in by_category.items():
        shuffled = list(items)
        random.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_ratio))
        test_items = shuffled[:n_test]
        train_items = shuffled[n_test:]

        test_texts.extend(test_items)
        test_labels.extend([cat] * len(test_items))
        train_texts.extend(train_items)
        train_labels.extend([cat] * len(train_items))

    return train_texts, train_labels, test_texts, test_labels


def train_and_evaluate(
    dataset_path: str = DATASET_PATH,
    model_output_path: str = MODEL_SAVE_PATH,
) -> NoteClassifier:
    """Run full training and evaluation workflow."""
    print("=" * 60)
    print(" Smart Notes Assistant — ML Training Pipeline")
    print("=" * 60)

    print(f"[*] Loading training dataset from: {dataset_path}")
    texts, labels = load_dataset(dataset_path)
    print(f"    Loaded {len(texts)} samples across {len(set(labels))} categories.")

    print("[*] Performing stratified 80/20 train/test split...")
    X_train, y_train, X_test, y_test = stratified_split(texts, labels, test_ratio=0.2, seed=42)
    print(f"    Training samples: {len(X_train)}")
    print(f"    Testing samples:  {len(X_test)}")

    print("[*] Training TF-IDF Vectorizer + Logistic Regression Classifier...")
    clf = NoteClassifier()
    clf.fit(X_train, y_train)
    print("    Training completed successfully.")

    print("\n[*] Evaluating model on held-out test set...")
    metrics = clf.evaluate(X_test, y_test)

    print("-" * 60)
    print(f" Model Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f" Macro Precision: {metrics['precision'] * 100:.2f}%")
    print(f" Macro Recall:    {metrics['recall'] * 100:.2f}%")
    print(f" Macro F1 Score:  {metrics['macro_f1'] * 100:.2f}%")
    print("-" * 60)

    print("\n[*] Per-Class Metrics:")
    for cat, m in metrics["per_class_metrics"].items():
        print(
            f"  {cat:<12} | Precision: {m['precision']:.2f} | "
            f"Recall: {m['recall']:.2f} | F1: {m['f1_score']:.2f} | Support: {m['support']}"
        )

    print("\n[*] Confusion Matrix (Rows = Actual, Columns = Predicted):")
    cats = metrics["categories"]
    header = "             " + " ".join(f"{c[:4]:>5}" for c in cats)
    print(header)
    for i, row in enumerate(metrics["confusion_matrix"]):
        row_str = " ".join(f"{v:>5}" for v in row)
        print(f"  {cats[i]:<10} {row_str}")

    print(f"\n[*] Saving trained model to: {model_output_path}")
    clf.save(model_output_path)
    print("    Model successfully saved and ready for inference!")
    print("=" * 60)

    return clf


if __name__ == "__main__":
    train_and_evaluate()
