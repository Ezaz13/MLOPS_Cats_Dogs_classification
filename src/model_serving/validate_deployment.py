import requests
import json
import time

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
API_URL = "http://127.0.0.1:5000/predict"


# ------------------------------------------------------------------
# Test Datasets
# ------------------------------------------------------------------
# These datasets use the raw values expected by the HTML form.
# The app.py logic handles mapping these to the model's expected format.
datasets = [
    # 1. Healthy Female (Low Risk)
    # Age 41, Female, Atypical Angina, Normal BP/Chol, No Risk Factors
    {
        "age": 41, "sex": 0, "cp": 1, "trestbps": 130, "chol": 204,
        "fbs": 0, "restecg": 0, "thalach": 172, "exang": 0,
        "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 1
    },
    # 2. High Risk Male (Likely Positive)
    # Age 63, Male, Asymptomatic, High BP, High Chol, FBS>120, ST Depression
    {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 2, "ca": 0, "thal": 2
    },
    # 3. Severe Risk (Angina + Depression + Reversible Defect)
    {
        "age": 57, "sex": 1, "cp": 3, "trestbps": 165, "chol": 289,
        "fbs": 0, "restecg": 1, "thalach": 124, "exang": 1,
        "oldpeak": 1.0, "slope": 1, "ca": 3, "thal": 3
    },
    # 4. Young Healthy Male
    {
        "age": 29, "sex": 1, "cp": 1, "trestbps": 120, "chol": 200,
        "fbs": 0, "restecg": 0, "thalach": 180, "exang": 0,
        "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 1
    },
    # 5. Elderly Female with High BP
    {
        "age": 70, "sex": 0, "cp": 2, "trestbps": 150, "chol": 300,
        "fbs": 0, "restecg": 2, "thalach": 130, "exang": 1,
        "oldpeak": 1.5, "slope": 1, "ca": 2, "thal": 3
    },
    # 6. High Cholesterol but otherwise okay
    {
        "age": 50, "sex": 1, "cp": 2, "trestbps": 135, "chol": 400,
        "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0,
        "oldpeak": 0.5, "slope": 0, "ca": 0, "thal": 1
    },
    # 7. Diabetic Risk (High FBS)
    {
        "age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 240,
        "fbs": 1, "restecg": 0, "thalach": 145, "exang": 1,
        "oldpeak": 1.2, "slope": 1, "ca": 1, "thal": 3
    },
    # 8. Typical Angina Case
    {
        "age": 48, "sex": 0, "cp": 0, "trestbps": 130, "chol": 250,
        "fbs": 0, "restecg": 1, "thalach": 155, "exang": 1,
        "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 1
    },
    # 9. Bradycardia & Depression (Low HR)
    {
        "age": 65, "sex": 1, "cp": 3, "trestbps": 160, "chol": 220,
        "fbs": 0, "restecg": "ab", "thalach": 110, "exang": 1,
        "oldpeak": 3.0, "slope": 2, "ca": 2, "thal": 2
    },
    # 10. Average Middle-Aged Male
    {
        "age": 54, "sex": 1, "cp": 2, "trestbps": 135, "chol": 250,
        "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 0.5, "slope": 1, "ca": 1, "thal": 3
    }
]

def run_tests():
    print(f"\nTesting Model Serving at {API_URL}...")
    print("=" * 75)
    print(f"{'Case':<5} | {'Prediction':<15} | {'Confidence':<12} | {'Status':<10}")
    print("-" * 75)

    for i, data in enumerate(datasets, 1):
        try:
            response = requests.post(API_URL, json=data)
            
            if response.status_code == 200:
                result = response.json()[0]
                pred_label = "High Risk" if result['prediction'] == 1 else "Low Risk"
                conf_label = f"{result['confidence']:.2%}"
                status = " OK"
            else:
                pred_label = "N/A"
                conf_label = "N/A"
                status = f"{response.status_code}"
                
            print(f"{i:<5} | {pred_label:<15} | {conf_label:<12} | {status:<10}")
            
        except requests.exceptions.ConnectionError:
            print(f"\n Error: Could not connect to {API_URL}.")
            print("   Make sure the Flask app is running (python src/model_serving/app.py)")
            break
        except Exception as e:
            print(f"{i:<5} | Error: {str(e)}")

if __name__ == "__main__":
    run_tests()