import os
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# -----------------------------
# File paths
# -----------------------------
TRAIN_FILE_PATH = "structured_analytics/data/processed/high_risk_train_data.csv"
TEST_FILE_PATH = "structured_analytics/data/processed/high_risk_test_data.csv"

MODELS_DIR = "structured_analytics/models"
METRICS_DIR = "structured_analytics/outputs/metrics"
PLOTS_DIR = "structured_analytics/outputs/plots"

TARGET_COLUMN = "high_clinical_risk"

CATEGORICAL_FEATURES = ["sex"]
NUMERICAL_FEATURES = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "glucose",
    "cholesterol",
    "creatinine",
    "diabetes",
    "hypertension"
]


def create_directories():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_data():
    train_df = pd.read_csv(TRAIN_FILE_PATH)
    test_df = pd.read_csv(TEST_FILE_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor


def build_models(preprocessor):
    models = {
        "highrisk_logistic_regression": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                random_state=42,
                max_iter=1000
            ))
        ]),

        "highrisk_random_forest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                n_jobs=1
            ))
        ]),

        "highrisk_logreg_smote": ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", LogisticRegression(
                random_state=42,
                max_iter=1000
            ))
        ]),

        "highrisk_xgboost_smote": ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", XGBClassifier(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42
            ))
        ])
    }

    return models


def evaluate_model(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob)
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n==============================")
    print(f"Model: {model_name}")
    print(f"==============================")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    metrics_file = os.path.join(METRICS_DIR, f"{model_name}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    report_file = os.path.join(METRICS_DIR, f"{model_name}_classification_report.txt")
    with open(report_file, "w") as f:
        f.write(report)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    cm_plot_file = os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_plot_file)
    plt.close()

    model_file = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    joblib.dump(model, model_file)

    return metrics


def save_summary(all_metrics):
    summary_df = pd.DataFrame(all_metrics).T
    summary_csv = os.path.join(METRICS_DIR, "high_risk_model_summary.csv")
    summary_txt = os.path.join(METRICS_DIR, "high_risk_model_summary.txt")

    summary_df.to_csv(summary_csv)

    with open(summary_txt, "w") as f:
        f.write(summary_df.to_string())

    print("\nHigh-risk model comparison summary:")
    print(summary_df)


def main():
    create_directories()

    X_train, X_test, y_train, y_test = load_data()
    preprocessor = build_preprocessor()
    models = build_models(preprocessor)

    all_metrics = {}

    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model_name, model, X_test, y_test)
        all_metrics[model_name] = metrics

    save_summary(all_metrics)


if __name__ == "__main__":
    main()