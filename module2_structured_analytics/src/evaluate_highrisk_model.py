import os
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# -----------------------------
# File paths
# -----------------------------
MODEL_PATH = "structured_analytics/models/highrisk_logistic_regression.pkl"
TEST_FILE_PATH = "structured_analytics/data/processed/high_risk_test_data.csv"
OUTPUT_PATH = "structured_analytics/outputs/metrics/highrisk_threshold_analysis.csv"

TARGET_COLUMN = "high_clinical_risk"

THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]


def load_model():
    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)
    return model


def load_test_data():
    print("Loading test data...")
    df = pd.read_csv(TEST_FILE_PATH)

    X_test = df.drop(columns=[TARGET_COLUMN])
    y_test = df[TARGET_COLUMN]

    return X_test, y_test


def evaluate_thresholds(model, X_test, y_test):
    print("\nEvaluating thresholds...")

    # Get probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    results = []

    for threshold in THRESHOLDS:
        y_pred = (y_prob >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        cm = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = cm.ravel()

        results.append({
            "threshold": threshold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        })

        print(f"\nThreshold: {threshold}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)

    results_df = pd.DataFrame(results)
    return results_df


def save_results(results_df):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved threshold analysis to:")
    print(OUTPUT_PATH)

    print("\nFinal comparison table:")
    print(results_df)


def main():
    model = load_model()
    X_test, y_test = load_test_data()

    results_df = evaluate_thresholds(model, X_test, y_test)
    save_results(results_df)


if __name__ == "__main__":
    main()