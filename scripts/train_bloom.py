"""Training and Comparative Evaluation Benchmark for Bloom's Taxonomy Classifiers.

Course: Natural Language Processing for LLM (24AM7PCLLM)
Institution: B.M.S. College of Engineering (BMSCE)
Department: Artificial Intelligence & Machine Learning

This script:
1. Loads data/bloom_dataset.csv (210 stratified university exam questions).
2. Performs an 80/20 stratified train-test split (seed=42).
3. Evaluates four approaches:
   - Approach A: Rule-Based (Action Verb keyword heuristic)
   - Approach B: TF-IDF + Logistic Regression
   - Approach C: Sentence-BERT (all-MiniLM-L6-v2) + Logistic Regression
   - Approach D: Zero-Shot LLM (Ollama llama3)
4. Computes Accuracy, Macro Precision, Macro Recall, and Macro F1-score.
5. Saves comparative results to results/bloom_comparison.csv.
6. Renders and saves a confusion matrix visualization to results/best_model_confusion_matrix.png.
7. Serializes the best-performing model to data/bloom_classifier_model.joblib.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.bloom_classifier import (
    RuleBasedBloomClassifier,
    TFIDFBloomClassifier,
    EmbeddingBloomClassifier,
    ZeroShotLLMBloomClassifier,
    BLOOM_LEVELS,
    MODEL_SAVE_PATH
)

DATA_PATH = "data/bloom_dataset.csv"
RESULTS_DIR = "results"
COMPARISON_CSV = os.path.join(RESULTS_DIR, "bloom_comparison.csv")
CONFUSION_MATRIX_PNG = os.path.join(RESULTS_DIR, "best_model_confusion_matrix.png")


def main():
    print("=" * 70)
    print("BLOOM'S TAXONOMY CLASSIFIER BENCHMARK (Course: 24AM7PCLLM)")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please generate data/bloom_dataset.csv first.")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} labelled questions from {DATA_PATH}.")
    print("Class distribution:")
    print(df["level"].value_counts().to_string())

    # 1. Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["question"].tolist(),
        df["level"].tolist(),
        test_size=0.20,
        random_state=42,
        stratify=df["level"].tolist()
    )
    print(f"\nSplit into {len(X_train)} training samples and {len(X_test)} testing samples.")

    models = {
        "Rule-Based (Keywords)": RuleBasedBloomClassifier(),
        "TF-IDF + Logistic Regression": TFIDFBloomClassifier(),
        "Sentence-BERT + Logistic Regression": EmbeddingBloomClassifier(),
        "Zero-Shot LLM (Ollama)": ZeroShotLLMBloomClassifier()
    }

    results = []
    predictions_dict = {}

    for name, model in models.items():
        print(f"\nEvaluating: {name}...")
        # Train if model requires fitting
        if hasattr(model, "fit"):
            model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict_batch(X_test)
        predictions_dict[name] = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=BLOOM_LEVELS, average="macro", zero_division=0
        )

        results.append({
            "Approach": name,
            "Accuracy": round(acc, 4),
            "Macro_Precision": round(prec, 4),
            "Macro_Recall": round(rec, 4),
            "Macro_F1": round(f1, 4)
        })
        print(f"  -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    # 2. Tabulate & Save Results
    results_df = pd.DataFrame(results)
    results_df.sort_values(by="Macro_F1", ascending=False, inplace=True)
    results_df.to_csv(COMPARISON_CSV, index=False)
    print("\n" + "=" * 70)
    print("COMPARATIVE EVALUATION SUMMARY:")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\nSaved comparison metrics to: {COMPARISON_CSV}")

    # 3. Identify Best Model & Save
    best_approach_name = results_df.iloc[0]["Approach"]
    best_model = models[best_approach_name]
    best_preds = predictions_dict[best_approach_name]
    print(f"\nBest Model: {best_approach_name} (F1: {results_df.iloc[0]['Macro_F1']})")

    # Serialize best model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"Saved best model object to: {MODEL_SAVE_PATH}")

    # 4. Generate & Save Confusion Matrix for the Best Model
    cm = confusion_matrix(y_test, best_preds, labels=BLOOM_LEVELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=BLOOM_LEVELS,
        yticklabels=BLOOM_LEVELS,
        cbar=True
    )
    plt.title(f"Confusion Matrix: {best_approach_name}\n(Academic Evaluation on 42 Test Questions)", fontsize=12, pad=12)
    plt.xlabel("Predicted Bloom Level", fontsize=11)
    plt.ylabel("Ground Truth Level", fontsize=11)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {CONFUSION_MATRIX_PNG}")
    print("=" * 70)


if __name__ == "__main__":
    main()

