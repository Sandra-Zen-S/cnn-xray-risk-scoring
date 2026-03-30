Structured Clinical Risk Module

Purpose:
This module takes structured clinical variables as input and returns a probability-based risk score, a triage priority label, and key contributing factors.

Input fields required:
- age
- sex
- bmi
- systolic_bp
- diastolic_bp
- glucose
- cholesterol
- creatinine
- diabetes
- hypertension

Example input:
{
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

Example output:
{
  "risk_score": 0.656,
  "priority": "HIGH",
  "top_factors": ["hypertension", "diabetes", "creatinine"]
}

Priority mapping:
- LOW: risk_score < 0.30
- MEDIUM: 0.30 <= risk_score < 0.50
- HIGH: risk_score >= 0.50
