import os
import pathlib
import logging
import sys
import pandas as pd
from flask import Flask, request, jsonify, render_template
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from flask import Response

# -------------------------------------------------
# Paths & MLflow (DB BACKEND ONLY CHANGE)
# -------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]

# Use environment variable if set (useful for Docker), otherwise fallback to local sqlite
database_path = BASE_DIR / "mlflow.db"
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{database_path}")
mlflow.set_tracking_uri(tracking_uri)

MODEL_URI = "models:/HeartDiseaseModel/latest"

# -------------------------------------------------
# Flask App
# -------------------------------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")
model = None
metrics = PrometheusMetrics(app)
# Categorical features used during training
CATEGORICAL_FEATURES = ["thal"]


# -------------------------------------------------
# Load model
# -------------------------------------------------
def load_model():
    global model
    try:
        app.logger.info(f"Attempting to load model from URI: {MODEL_URI}")
        model = mlflow.sklearn.load_model(MODEL_URI)
        app.logger.info(f"Successfully loaded model from {MODEL_URI}")
    except Exception as e:
        app.logger.warning(f"Standard load failed: {e}. Attempting path correction for Docker...")
        try:
            client = MlflowClient()
            # Parse model name from URI "models:/HeartDiseaseModel/latest"
            model_name = MODEL_URI.split("/")[1]
            app.logger.info(f"Querying MLflow Registry for model: {model_name}")

            # Get latest version info
            versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
            app.logger.info(f"Versions found: {[(v.version, v.current_stage, v.run_id) for v in versions]}")

            if not versions:
                raise Exception("No versions found")
            latest_version = max(versions, key=lambda x: int(x.version))
            app.logger.info(
                f"Selected Version: {latest_version.version} (Stage: {latest_version.current_stage}) Run ID: {latest_version.run_id}")

            # Fix path: Replace host absolute path with Docker container path
            source = latest_version.source
            app.logger.info(f"Original source path from registry: {source}")

            # Fallback: If source is abstract (e.g. models:/...), fetch the physical path from the run
            if "mlruns" not in source and "models:/" in source:
                app.logger.info(f"Source URI is abstract. Fetching run info...")
                run = client.get_run(latest_version.run_id)
                source = run.info.artifact_uri
                app.logger.info(f"Resolved run artifact URI: {source}")

            # Robust path correction for Docker (Windows host -> Linux container)
            # 1. Normalize slashes and remove file:// prefix
            source_norm = source.replace("\\", "/").replace("file://", "")

            docker_path = None
            idx = source_norm.find("mlruns")
            if idx != -1:
                # Extract everything after 'mlruns' and join with container path
                rel_path = source_norm[idx + 6:].lstrip("/")
                docker_path = os.path.join("/app/mlruns", rel_path).rstrip("/.")

            # Robust Search: Find the actual directory containing 'MLmodel'
            final_model_path = None

            # Strategy 1: Check if the constructed path or its subdirectories contain the model
            if docker_path and os.path.exists(docker_path):
                for root, dirs, files in os.walk(docker_path):
                    if "MLmodel" in files:
                        final_model_path = root
                        break

            # Strategy 2: Search for any directory containing 'MLmodel' that matches this run
            if not final_model_path:
                app.logger.info(f"Searching /app/mlruns for MLmodel...")
                for root, dirs, files in os.walk("/app/mlruns"):
                    if "MLmodel" in files and (latest_version.run_id in root or "models" in root):
                        final_model_path = root
                        break

            if final_model_path:
                app.logger.info(f"Loading model from found path: {final_model_path}")
                model = mlflow.sklearn.load_model(final_model_path)
            else:
                raise Exception(f"Could not locate MLmodel file for run {latest_version.run_id} in /app/mlruns")

        except Exception as e2:
            model = None
            app.logger.error(f"Failed to load model: {e2}")


# -------------------------------------------------
# UI
# -------------------------------------------------

@app.route("/metrics")
def metrics_endpoint():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


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

    app.logger.info("Starting inference...")
    app.logger.info(f"Received prediction parameters: {data}")

    try:
        df = pd.DataFrame([data])

        # ---------------------------------------------------------
        # 1. Data Mapping (Form values -> Model training standards)
        # ---------------------------------------------------------
        if 'cp' in df.columns:
            df['cp'] = df['cp'].map({0: 1, 1: 2, 2: 3, 3: 4})

        if 'slope' in df.columns:
            df['slope'] = df['slope'].map({0: 1, 1: 2, 2: 3})

        if 'thal' in df.columns:
            df['thal'] = df['thal'].map({1: 3, 2: 6, 3: 7})

        # ---------------------------------------------------------
        # 2. Feature Engineering
        # ---------------------------------------------------------
        df['rate_pressure_product'] = df['trestbps'] * df['thalach']
        df['chol_fbs_interaction'] = df['chol'] * df['fbs']

        # ---------------------------------------------------------
        # 3. One-Hot Encoding & Alignment
        # ---------------------------------------------------------
        cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        df_processed = pd.get_dummies(df, columns=cat_cols)

        if hasattr(model, "feature_names_in_"):
            df_processed = df_processed.reindex(
                columns=model.feature_names_in_,
                fill_value=0
            )

        preds = model.predict(df_processed)
        probs = model.predict_proba(df_processed)[:, 1]

        app.logger.info(f"Prediction: {int(preds[0])}, Confidence: {float(probs[0])}")

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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

app.logger.setLevel(logging.INFO)
# -------------------------------------------------
# Entrypoint
# -------------------------------------------------
if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=True)