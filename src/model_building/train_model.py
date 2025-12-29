"""
Model Training & Experiment Tracking using MLflow
Heart Disease Prediction Project

- Loads transformed dataset
- Builds preprocessing + model pipeline
- Trains Random Forest classifier
- Logs parameters, metrics, artifacts
- Registers model in MLflow Model Registry
"""

import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "transformed" / "transformed_heart_data.csv"
MLRUNS_PATH = PROJECT_ROOT / "mlruns"

# ------------------------------------------------------------------
# MLflow config
# ------------------------------------------------------------------
EXPERIMENT_NAME = "Heart Disease Prediction"
REGISTERED_MODEL_NAME = "HeartDiseaseModel"

mlflow.set_tracking_uri(f"file:///{MLRUNS_PATH.as_posix()}")
mlflow.set_experiment(EXPERIMENT_NAME)

# ------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------
def prepare_data(df: pd.DataFrame):
    if "target" not in df.columns:
        raise ValueError("Dataset must contain 'target' column")

    df = df.copy()
    df["target"] = (df["target"] > 0).astype(int)

    X = df.drop(columns=["target"])
    y = df["target"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------
def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "category"]).columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def main():
    print("Starting MLflow training pipeline...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = prepare_data(df)
    pipeline = build_pipeline(X_train)

    with mlflow.start_run(run_name="RandomForest"):

        # Train
        pipeline.fit(X_train, y_train)

        # Evaluate
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
        }

        mlflow.log_metrics(metrics)

        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": 200,
            "random_state": 42,
        })

        # ✅ Correct way to log & register model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train.head(5)
        )

        print("Model trained and registered successfully")
        print("Metrics:", metrics)

    print("Training pipeline completed")

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
