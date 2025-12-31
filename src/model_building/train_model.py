"""
End-to-End Model Training, Evaluation & Experiment Tracking
Heart Disease Prediction Project (MLOps)

Features:
- Unified preprocessing pipeline
- Multiple model comparison
- Stratified cross-validation
- MLflow experiment tracking (SQLite backend)
- Automatic markdown report generation
- Best model registration
"""

import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from datetime import datetime

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
REPORT_PATH = PROJECT_ROOT / "src" / "model_building" / "model_performance_report.md"

# ------------------------------------------------------------------
# MLflow Configuration
# ------------------------------------------------------------------
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Heart Disease Prediction"
REGISTERED_MODEL_NAME = "HeartDiseaseModel"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
# ------------------------------------------------------------------
# Data Preparation
# ------------------------------------------------------------------
def prepare_data(df: pd.DataFrame):
    if "target" not in df.columns:
        raise ValueError("Dataset must contain a 'target' column")

    df = df.copy()
    df["target"] = (df["target"] > 0).astype(int)

    X = df.drop(columns=["target"])
    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

# ------------------------------------------------------------------
# Pipeline Builder
# ------------------------------------------------------------------
def build_pipeline(X: pd.DataFrame, classifier) -> Pipeline:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

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

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

# ------------------------------------------------------------------
# Markdown Report Generator
# ------------------------------------------------------------------
def generate_markdown_report(results: dict):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w") as f:
        f.write("# Model Performance Report\n\n")
        f.write(f"*Report generated on: {datetime.now():%Y-%m-%d %H:%M:%S}*\n\n")
        f.write("*MLflow Experiment: 'Heart Disease Prediction'*\n\n")

        for model, data in results.items():
            f.write(f"## {model}\n\n")
            f.write(f"- **MLflow Run ID**: `{data['run_id']}`\n")
            f.write(f"- **Best Params**: `{data['params']}`\n")
            f.write(f"- **CV ROC-AUC (Mean)**: {data['cv_roc_auc']:.4f}\n")
            f.write(f"- **Test Accuracy**: {data['accuracy']:.4f}\n")
            f.write(f"- **Precision**: {data['precision']:.4f}\n")
            f.write(f"- **Recall**: {data['recall']:.4f}\n")
            f.write(f"- **F1-Score**: {data['f1']:.4f}\n")
            f.write(f"- **ROC-AUC**: {data['roc_auc']:.4f}\n\n")

# ------------------------------------------------------------------
# Training & Model Selection
# ------------------------------------------------------------------
def main():
    print("Starting training pipeline...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_data(df)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = [
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(
                C=0.1,
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ),
            "params": {"C": 0.1, "solver": "liblinear"}
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "params": {"n_estimators": 100, "max_depth": 10}
        },
        {
            "name": "Gradient Boosting",
            "estimator": GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ),
            "params": {"n_estimators": 150, "learning_rate": 0.05}
        }
    ]

    results = {}
    best_score = 0.0
    best_run_id = None
    best_model_name = ""

    for model in models:
        name = model["name"]
        print(f"Training {name}...")

        with mlflow.start_run(run_name=name) as run:
            pipeline = build_pipeline(X_train, model["estimator"])

            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1
            )

            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_test)
            probs = pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                "cv_roc_auc": cv_scores.mean(),
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds),
                "recall": recall_score(y_test, preds),
                "f1": f1_score(y_test, preds),
                "roc_auc": roc_auc_score(y_test, probs),
            }

            mlflow.log_metrics(metrics)
            mlflow.log_params(model["params"])
            mlflow.set_tags({
                "domain": "healthcare",
                "problem_type": "binary_classification"
            })

            # Log & Register model
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                input_example=X_train.head(5)
            )

            results[name] = {
                **metrics,
                "params": model["params"],
                "run_id": run.info.run_id
            }

            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                best_run_id = run.info.run_id
                best_model_name = name

    generate_markdown_report(results)

    if best_run_id:
        mlflow.register_model(
            model_uri=f"runs:/{best_run_id}/model",
            name=REGISTERED_MODEL_NAME
        )
        print(f"Best model registered: {best_model_name}")

    print("Training pipeline completed successfully.")

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
