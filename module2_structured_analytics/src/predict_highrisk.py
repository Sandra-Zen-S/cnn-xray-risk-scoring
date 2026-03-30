import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "structured_analytics/models/highrisk_logistic_regression.pkl"

REQUIRED_FEATURES = [
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

TOP_FACTOR_MAP = {
    "num__hypertension": "hypertension",
    "num__diabetes": "diabetes",
    "num__creatinine": "creatinine",
    "num__age": "age",
    "num__cholesterol": "cholesterol",
    "num__bmi": "bmi",
    "num__diastolic_bp": "diastolic_bp",
    "num__glucose": "glucose",
    "num__systolic_bp": "systolic_bp",
    "cat__sex_Female": "sex_female",
    "cat__sex_Male": "sex_male",
    "cat__sex_Other": "sex_other"
}


def load_model():
    model = joblib.load(MODEL_PATH)
    return model


def get_priority(score: float) -> str:
    if score >= 0.50:
        return "HIGH"
    elif score >= 0.30:
        return "MEDIUM"
    else:
        return "LOW"


def get_top_factors(model, top_n=3):
    """
    Returns top clinically meaningful global factors.
    Excludes one-hot encoded sex variables to keep the explanation cleaner.
    """
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients)
    })

    # Exclude sex dummy variables from explanation
    coef_df = coef_df[~coef_df["feature"].str.startswith("cat__sex_")]

    coef_df = coef_df.sort_values(by="abs_coefficient", ascending=False)

    top_features = []
    for feat in coef_df["feature"].head(top_n):
        top_features.append(TOP_FACTOR_MAP.get(feat, feat))

    return top_features


def predict_structured_risk(input_data: dict):
    """
    input_data should contain:
    age, sex, bmi, systolic_bp, diastolic_bp, glucose,
    cholesterol, creatinine, diabetes, hypertension
    """

    missing = [col for col in REQUIRED_FEATURES if col not in input_data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    model = load_model()

    input_df = pd.DataFrame([input_data], columns=REQUIRED_FEATURES)

    risk_score = float(model.predict_proba(input_df)[0][1])
    priority = get_priority(risk_score)
    top_factors = get_top_factors(model, top_n=3)

    result = {
        "risk_score": round(risk_score, 4),
        "priority": priority,
        "top_factors": top_factors
    }

    return result


if __name__ == "__main__":
    sample_input = {
        "age": 68,
        "sex": "Male",
        "bmi": 29.5,
        "systolic_bp": 145.0,
        "diastolic_bp": 92.0,
        "glucose": 160.0,
        "cholesterol": 220.0,
        "creatinine": 1.4,
        "diabetes": 1,
        "hypertension": 1
    }

    prediction = predict_structured_risk(sample_input)
    print("\nStructured Risk Prediction Output:")
    print(prediction)