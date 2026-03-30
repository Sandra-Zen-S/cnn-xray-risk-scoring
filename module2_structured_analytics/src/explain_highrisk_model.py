import os
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "structured_analytics/models/highrisk_logistic_regression.pkl"
OUTPUT_DIR = "structured_analytics/outputs/metrics"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = joblib.load(MODEL_PATH)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients)
    })

    coef_df = coef_df.sort_values(by="abs_coefficient", ascending=False)

    print("\nFeature Importance for Final High-Risk Logistic Regression Model:")
    print(coef_df[["feature", "coefficient"]])

    output_file = os.path.join(OUTPUT_DIR, "highrisk_logistic_feature_importance.csv")
    coef_df.to_csv(output_file, index=False)

    print(f"\nSaved feature importance to: {output_file}")


if __name__ == "__main__":
    main()