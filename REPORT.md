# MLOps Assignment Report
# Title: Heart Disease UCI Dataset

# Group 130

| Sl. No. | Name | BITS ID | Contribution |
| :--- | :--- | :--- | :--- |
| 1 | MD. EZAZUL HAQUE | 2024aa05083 | 100% |
| 2 | SWAPNIL BHUSHAN VERMA | 2024ab05216 | 100% |
| 3 | MAYANK SPARSH | 2024aa05386 | 100% |
| 4 | MD. SHAFI HUSSAIN | 2024ab05039 | 100% |
| 5 | M. MOHIT SHARMA | 2023ac05887 | 100% |


## 1. Setup/Install Instructions

### Prerequisites
-   **OS**: Windows, macOS, or Linux.
-   **Tools**: Python 3.9+, Git, Docker Desktop.

### Step-by-Step Guide
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Ezaz13/MLOPS-Heart-Disease.git
    cd Heart-Disease-Prediction-Project
    ```

2.  **Environment Setup**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Unix
    source venv/bin/activate
    ```

3.  **Install Requirements**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run End-to-End Pipeline**
    Execute the full pipeline (Ingestion -> Validation -> Preparation -> Transformation -> Model Building):
    ```bash
    python src/data_pipeline_orchestrator/pipeline.py
    ```
    **Pipeline Flow**:
    The script executes the following DAG sequentially, waiting for each task to complete:
    1.  **Data Ingestion** (`src/data_ingestion/ingestion.py`):
        -   Downloads the dataset directly from the UCI Machine Learning Repository URL.
        -   Implements retry logic (3 attempts) to handle network instability.
        -   Saves the raw CSV to `data/raw/uci` with a timestamp.
    2.  **Data Validation** (`src/data_validation/validation.py`):
        -   Uses **Great Expectations** to validate the downloaded data against a defined schema.
        -   Checks column order, non-null constraints for critical fields (`age`, `sex`, `target`), and value ranges (e.g., `age` between 20-100).
        -   Generates validation reports and raises an error if integrity checks fail.
    3.  **Data Preparation** (`src/data_preparation/preparation.py`):
        -   Handles data cleaning (replacing '?' with NaN).
        -   Normalizes the target variable to binary (0/1).
        -   Generates comprehensive EDA reports (histograms, boxplots, correlation matrices).
        -   **Output Location**: All EDA artifacts are saved to `artifacts/eda/eda_run_<timestamp>/`.
        -   Performs initial preprocessing: Imputes missing values, scales numerical features (`StandardScaler`), and One-Hot Encodes categorical variables.
        -   Saves the processed dataset to `data/prepared`.
    4.  **Data Transformation** (`src/data_transformation/transformation.py`):
        -   Loads the prepared data and performs Feature Engineering.
        -   Creates derived features:
            -   **Rate Pressure Product**: `thalach` * `trestbps`.
            -   **Age Groups**: Quantile binning of age ranges.
            -   **High Risk Flag**: Composite indicator based on `oldpeak` and `ca` thresholds.
        -   Saves the augmented dataset to `data/transformed`.
    5.  **Model Building** (`src/model_building/train_model.py`):
        -   Loads the transformed dataset.
        -   Splits data into training and testing sets (Stratified Split).
        -   Defines a training pipeline (including redundant scaling/encoding for robustness).
        -   Trains/Evaluates three models (Logistic Regression, Random Forest, Gradient Boosting).
        -   Logs all params, metrics, and artifacts to **MLflow** and registers the best model.

5.  **Start API Server**
    ```bash
    python src/model_serving/app.py
    ```
7.  **Validate Deployment**
    Ensure the service is running correctly using the provided validation script or UI:
    -   **CLI Test (Batch)**: Run the batch validation script to test multiple scenarios against the API.
        ```bash
        python src/model_serving/validate_deployment.py
        ```
    -   **CLI Test (Single Request)**:
        ```bash
        curl -X POST http://localhost:5000/predict -d '{"age":63, "sex":1, "cp":3, "trestbps":145, "chol":233, "fbs":1, "restecg":0, "thalach":150, "exang":0, "oldpeak":2.3, "slope":0, "ca":0, "thal":1}'
        ```
    -   **UI Test**: Open a browser and navigate to:
        ```
        http://localhost:5000/
        ```
        This loads `index.html`, where you can manually input patient data and get real-time predictions.

## 2. EDA and Modelling Choices

### Exploratory Data Analysis (EDA)
The data preparation pipeline (`src/data_preparation/preparation.py`) generates comprehensive analysis reports to guide feature engineering.

-   **Artifact Location**: All EDA outputs are automatically saved to:
    `artifacts/eda/eda_run_<YYYYMMDD_HHMMSS>/`
-   **Generated Insights**:
    1.  **Summary Statistics**: saved as `summary_statistics.csv`.
    2.  **Distributions**: Histograms showing the distribution of numerical features (like `age`, `chol`) segmented by the target variable (Disease vs. No Disease).
    3.  **Outlier Detection**: Boxplots for numerical features to identify anomalies.
    4.  **Categorical Analysis**: Count plots for features like `cp` (chest pain) and `exang` (exercise angina).
    5.  **Correlation Analysis**: A heatmap (`numeric_correlation_heatmap.png`) visualizing relationships between features.
-   **Key Findings**:
    -   **Target Balance**: The dataset is balanced, eliminating the immediate need for SMOTE.
    -   **Predictive Features**: `cp` (Type 3) and `oldpeak` showed strong differentiation between positive/negative cases.

### Preprocessing Pipeline
To ensure robust model training, the following transformations are applied:
1.  **Missing Value Imputation**: Median for numerical columns, Mode for categorical.
2.  **Scaling**: `StandardScaler` to normalize numerical features (`age`, `chol`, etc.).
3.  **Encoding**: One-Hot Encoding for categorical variables.

### Modelling Methodology
We evaluated three diverse algorithms using **Stratified K-Fold Cross-Validation (K=5)** to ensure the model generalizes well to unseen data.

#### Model Performance Comparison

| Model | CV ROC-AUC (Mean) | Test Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.9009** | 0.8689 | 0.8125 | **0.9286** | 0.8667 |
| Random Forest | 0.8840 | **0.8852** | 0.8387 | 0.9286 | **0.8814** |
| Gradient Boosting | 0.8528 | 0.8689 | 0.8125 | 0.9286 | 0.8667 |


#### Selection Logic
The automated selection logic (defined in `src/model_building/train_model.py`) prioritizes **F1 Score** as the primary metric for stability, particularly for imbalanced datasets or costs of errors.

**Why F1 Score?**
The F1 Score is the harmonic mean of Precision and Recall. In the context of Heart Disease prediction:
-   **High Recall** is crucial: We must not miss any positive cases (False Negatives), as failing to diagnose a patient could be fatal.
-   **Precision matters**: We also want to minimize False Positives to avoid unnecessary treatment and anxiety.
F1 Score provides a single metric that balances these two competing objectives, ensuring the model is robust and reliable for medical diagnosis, rather than just being accurate on the majority class.

> **Selected Model: Random Forest**
>
> **Random Forest** was selected because it achieved the highest **F1 Score (0.8814)**. This balanced metric ensures that the model maintains a good trade-off between Precision and Recall, which is critical for medical diagnosis where both false positives and false negatives carry significant costs. Additionally, it maintained a high Test Accuracy (0.8852) and strong Recall (0.9286).

## 3. Experiment Tracking Summary

We successfully integrated **MLflow** to track the entire machine learning lifecycle, ensuring reproducibility and observability.

### 3.1 Configuration & Setup
-   **Backend Store**: SQLite (`sqlite:///mlflow.db`) for lightweight local metadata storage.
-   **Artifact Store**: Local directory `mlruns/` for storing models and plots.
-   **Experiment Name**: `Heart Disease Prediction`

### 3.2 Tracked Metrics & Parameters
For each training iteration, the following were logged:
-   **Hyperparameters**: Model-specific configs (e.g., Logistic Regression `C=0.1`, Random Forest `n_estimators=100`).
-   **Performance Metrics**:
    -   Primary: **CV ROC-AUC** (Used for selection).
    -   Secondary: Recall (Critical for healthcare), Precision, F1-Score, and Accuracy.
-   **Tags**: Metadata for searchability:
    -   `domain`: "healthcare"
    -   `problem_type`: "binary_classification"

### 3.3 Recorded Runs (Sample)
Based on the latest experiment cycle:

| Model | Run ID | Status | Key Result |
| :--- | :--- | :--- | :--- |
| **Random Forest** | `57ecee26e31d450d8e4f3e7ed35781ab` | **Registered** | **Highest F1 Score (0.8814)** |
| Logistic Regression | `1ba457450f2641a69d41a89a44639546` | Archived | F1 Score: 0.8667 |
| Gradient Boosting | `ea2881acdae34c1fad7cb07d78d0f4fb` | Archived | F1 Score: 0.8667 |

### 3.4 Model Registry & Artifacts
-   **Artifacts Preserved**:
    -   `model.pkl`: The serialized Scikit-learn pipeline.
    -   `MLmodel`: Metadata defining the model flavor and dependencies.
    -   `conda.yaml` / `requirements.txt`: Environment definitions for reproduction.
    -   `input_example.json`: A sample of the training data (first 5 rows) to validate schema during serving.

-   **Registration Strategy**:
    The training script automatically compares the `f1_score` of all candidates. The model with the highest score (Random Forest) was programmatically registered in the MLflow Model Registry as:
    -   **Model Name**: `HeartDiseaseModel`
    -   **Version**: `2`
    -   **Stage**: Ready for Production / Deployment.

## 4. Architecture Diagram

The system follows a microservices architecture pattern with offline training and online serving.

```mermaid
graph TD
    subgraph "Data Pipeline Orchestrator"
        Ingest[Data Ingestion<br/>(ingestion.py)] --> Validate[Data Validation<br/>(validation.py)]
        Validate --> Prep[Data Preparation<br/>(preparation.py)]
        Prep --> Transform[Data Transformation<br/>(transformation.py)]
        Transform --> Train[Model Building<br/>(train_model.py)]
    end

    subgraph "MLOps Infrastructure"
        Train -->|Log Metrics & Model| MLflow[MLflow Tracking<br/>(SQLite/Local)]
        MLflow -->|Register Best Model| Registry[Model Registry]
    end

    subgraph "Production Deployment"
        Registry -->|Load Model| App[Flask Service<br/>(app.py)]
        App -->|Containerize| Docker[Docker Image]
        Docker -->|Deploy| K8s[Kubernetes Cluster]
        K8s -->|Expose| API[REST API]
    end

    User[End User] -->|POST Predictions| API
```

## 5. CI/CD and Deployment Workflow

### CI/CD Pipeline
Managed via **GitHub Actions** (`.github/workflows/ci_cd_pipeline.yml`).
-   **Trigger**: Push or Pull Request to branches `main` or `master`.
-   **Jobs**:

    #### 1. Build, Test, and Train (`build-test-train`)
    -   **Environment**: Ubuntu runner with Python 3.9.
    -   **Linting**: Uses `flake8` to enforce PEP8 standards (max line length: 127).
    -   **Unit Testing**: Executes `pytest` on the `tests/` directory.
    -   **Pipeline Execution**: Runs `src/data_pipeline_orchestrator/pipeline.py` to execute data ingestion, validation, transformation, and model training.
    -   **Artifact Archival**: Uploads critical artifacts for subsequent jobs:
        -   `mlflow-runs` and `mlflow.db`: Preserves the trained model and tracking database.
        -   `model-performance-report`: Markdown summary.
        -   `validation-reports`: Data quality checks.
        -   `pipeline-logs`: Execution logs.

    #### 2. Docker Build & Push (`docker-build-push`)
    -   **Condition**: Runs only on `push` events.
    -   **Artifact Retrieval**: Downloads `mlflow-runs` and `mlflow.db` from the build job to include the trained model in the image.
    -   **Build**: Uses Docker Buildx to build the image from the `Dockerfile`.
    -   **Push**: Pushes the image to Docker Hub tagged with `latest` and the commit SHA.

    #### 3. Kubernetes Deployment Test (`deploy-k8s-kind`)
    -   **Condition**: Runs only on `push` events.
    -   **Environment**: Sets up a **Kind (Kubernetes in Docker)** cluster directly on the runner.
    -   **Deployment**:
        -   Pulls the newly built image.
        -   Loads the image into the Kind cluster.
        -   Applies `k8/deployment.yaml` and `k8/service.yaml`.
        -   Updates the deployment to use the specific image version (SHA).
    -   **Verification**: Waits for the rollout to complete (`kubectl rollout status`) and logs cluster status.

### Deployment Configuration
-   **Containerization** (`Dockerfile`):
    -   **Base Image**: `python:3.10-slim`.
    -   ** Configuration**: Sets `MLFLOW_TRACKING_URI` to `sqlite:////app/mlflow.db`.
    -   **Exposed Port**: 5000.
    -   **Entrypoint**: Starts the Flask application (`src/model_serving/app.py`).

-   **Orchestration** (`k8/` directory):
    -   **Deployment** (`deployment.yaml`): Manages application replicas for high availability and defines resource constraints.
    -   **Service** (`service.yaml`): Exposes the application to external traffic using a LoadBalancer strategy.
    -   **Monitoring** (`service-monitor.yaml`): Configures Prometheus scraping for real-time metrics.

## 6. Link to Code Repository

[https://github.com/Ezaz13/MLOPS-Heart-Disease.git](https://github.com/Ezaz13/MLOPS-Heart-Disease.git)


## 6. Link to Application walkthrough recording

https://wilpbitspilaniacin0-my.sharepoint.com/:v:/g/personal/2024aa05083_wilp_bits-pilani_ac_in/IQByKQZjA01pTZDbOenZWesFAWihpS3Pcps-bkGT9a37-tg?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=2EyaNt