import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_FILE_PATH = "structured_analytics/data/raw/clinical_data.csv"
PROCESSED_DIR = "structured_analytics/data/processed"

SELECTED_FEATURES = [
    "age",
    "sex",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "glucose",
    "cholesterol",
    "creatinine",
    "diabetes",
    "hypertension"
]

NEW_TARGET_COLUMN = "high_clinical_risk"

SEVERE_DIAGNOSES = ["Sepsis", "Heart Failure", "Pneumonia"]


def create_high_risk_target(df):
    df[NEW_TARGET_COLUMN] = (
        (df["mortality"] == 1) |
        (df["readmission_30d"] == 1) |
        (df["diagnosis"].isin(SEVERE_DIAGNOSES))
    ).astype(int)
    return df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = pd.read_csv(RAW_FILE_PATH)
    print("\nLoaded raw dataset successfully.")
    print("Original shape:", df.shape)

    df = create_high_risk_target(df)

    print(f"\nNew target column created: {NEW_TARGET_COLUMN}")
    print("\nHigh clinical risk distribution:")
    print(df[NEW_TARGET_COLUMN].value_counts())
    print(df[NEW_TARGET_COLUMN].value_counts(normalize=True))

    selected_columns = SELECTED_FEATURES + [NEW_TARGET_COLUMN]
    df_selected = df[selected_columns].copy()

    print("\nSelected columns:")
    print(df_selected.columns.tolist())

    X = df_selected[SELECTED_FEATURES]
    y = df_selected[NEW_TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_df = X_train.copy()
    train_df[NEW_TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df[NEW_TARGET_COLUMN] = y_test

    full_file = os.path.join(PROCESSED_DIR, "high_risk_structured_data.csv")
    train_file = os.path.join(PROCESSED_DIR, "high_risk_train_data.csv")
    test_file = os.path.join(PROCESSED_DIR, "high_risk_test_data.csv")

    df_selected.to_csv(full_file, index=False)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print("\nProcessed files saved successfully.")
    print(f"Full dataset: {full_file}")
    print(f"Train dataset: {train_file}")
    print(f"Test dataset: {test_file}")

    print("\nTrain shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTraining target distribution:")
    print(train_df[NEW_TARGET_COLUMN].value_counts())
    print(train_df[NEW_TARGET_COLUMN].value_counts(normalize=True))

    print("\nTest target distribution:")
    print(test_df[NEW_TARGET_COLUMN].value_counts())
    print(test_df[NEW_TARGET_COLUMN].value_counts(normalize=True))


if __name__ == "__main__":
    main()