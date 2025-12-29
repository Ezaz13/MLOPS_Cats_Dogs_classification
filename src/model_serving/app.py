import os
import pathlib
import pandas as pd
from flask import Flask, request, jsonify, render_template
import mlflow
import mlflow.sklearn

# -------------------------------------------------
# Paths & MLflow
# -------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
MLRUNS_PATH = BASE_DIR / "mlruns"

mlflow.set_tracking_uri(f"file:///{MLRUNS_PATH.as_posix()}")

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/HeartDiseaseModel/latest"
)

# -------------------------------------------------
# Flask App
# -------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
model = None

# Categorical features used during training
CATEGORICAL_FEATURES = ["thal"]


def load_model():
    global model
    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
        app.logger.info(f"Model loaded from {MODEL_URI}")
    except Exception as e:
        model = None
        app.logger.error(f"Failed to load model: {e}")


# -------------------------------------------------
# UI
# -------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# -------------------------------------------------
# Health
# -------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if model is not None else "error",
        "model_loaded": model is not None,
        "model_uri": MODEL_URI
    })


# -------------------------------------------------
# Prediction
# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON payload"}), 400

    try:
        df = pd.DataFrame([data])

        # ---------------------------------------------------------
        # 1. Data Mapping (Form values -> Model training standards)
        # ---------------------------------------------------------
        # Map 'cp' (0-3) to (1-4)
        if 'cp' in df.columns:
            df['cp'] = df['cp'].map({0: 1, 1: 2, 2: 3, 3: 4})
        
        # Map 'slope' (0-2) to (1-3)
        if 'slope' in df.columns:
            df['slope'] = df['slope'].map({0: 1, 1: 2, 2: 3})
            
        # Map 'thal' (1-3) to (3, 6, 7)
        if 'thal' in df.columns:
            df['thal'] = df['thal'].map({1: 3, 2: 6, 3: 7})

        # ---------------------------------------------------------
        # 2. Feature Engineering (Replicating transformation pipeline)
        # ---------------------------------------------------------
        # Calculate interaction terms expected by the model
        df['rate_pressure_product'] = df['trestbps'] * df['thalach']
        df['chol_fbs_interaction'] = df['chol'] * df['fbs']
        
        # Placeholder for 'is_high_risk' (logic unknown, defaulting to 0 to prevent crash)
        df['is_high_risk'] = 0

        # ---------------------------------------------------------
        # 3. One-Hot Encoding & Alignment
        # ---------------------------------------------------------
        # Convert categoricals to float to match column names like 'sex_1.0'
        cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        # Apply One-Hot Encoding
        df_processed = pd.get_dummies(df, columns=cat_cols)

        # Align columns with model (fill missing OHE columns with 0)
        if hasattr(model, "feature_names_in_"):
            df_processed = df_processed.reindex(columns=model.feature_names_in_, fill_value=0)

        preds = model.predict(df_processed)
        probs = model.predict_proba(df_processed)[:, 1]

        return jsonify([{
            "prediction": int(preds[0]),
            "confidence": float(probs[0])
        }])

    except Exception as e:
        app.logger.error(f"Inference failed: {e}")
        return jsonify({
            "error": "Inference failed",
            "details": str(e)
        }), 400


# -------------------------------------------------
# Entrypoint
# -------------------------------------------------
if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=9696, debug=True)
