# module3_fusion/fusion.py
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Module 1
from module1_imaging.predict import predict_image

# Import Module 2
from module2_structured_analytics.src.predict_highrisk import predict_structured_risk


def compute_final_risk(imaging_output, structured_output):

    imaging_score = imaging_output["risk_score"]
    structured_score = structured_output["risk_score"]

    # Fusion logic
    final_score = (0.6 * imaging_score) + (0.4 * structured_score)

    # Final priority
    if final_score >= 0.7:
        final_priority = "HIGH"
    elif final_score >= 0.4:
        final_priority = "MEDIUM"
    else:
        final_priority = "LOW"

    return {
        "final_score": round(final_score, 4),
        "final_priority": final_priority,
        "imaging_score": imaging_score,
        "structured_score": structured_score,
        "top_factors": structured_output.get("top_factors", [])
    }


# 🔥 NEW FUNCTION (Module 4)
def map_to_tier(priority):
    if priority == "HIGH":
        return "Tier 1 - Immediate Attention"
    elif priority == "MEDIUM":
        return "Tier 2 - Urgent"
    else:
        return "Tier 3 - Standard"


if __name__ == "__main__":

    # 1. Image input
    image_path = "module1_imaging/data/images/sample/sample/images/00000013_005.png"
    imaging_output = predict_image(image_path)

    # 2. Clinical input
    patient_data = {
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

    structured_output = predict_structured_risk(patient_data)

    # 3. Fusion
    result = compute_final_risk(imaging_output, structured_output)

    # 4. Map to ED Tier
    tier = map_to_tier(result["final_priority"])

    # 5. Display output (FINAL DEMO)
    print("\n===== Emergency Department Triage =====")

    print(f"\nFinal Priority: {result['final_priority']}")
    print(f"ED Priority Tier: {tier}")
    print(f"Final Score: {result['final_score']}")

    print(f"\nImaging Score: {result['imaging_score']}")
    print(f"Structured Score: {result['structured_score']}")

    print("\nTop Risk Factors:")
    for factor in result["top_factors"]:
        print(f"- {factor}")

    print("\nReason for Flagging:")
    for factor in result["top_factors"]:
        print(f"- High {factor}")
