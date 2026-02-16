# MLOps Assignment Report
# Title: Cats vs Dogs Classification (Computer Vision)

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
-   **Hardware**: GPU recommended (CUDA) for faster training, but CPU is supported.

### Step-by-Step Guide
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Ezaz13/MLOPS_Cats_Dogs_classification.git
    cd Cats_Dogs_Classification-Project
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
    # For development (full dependencies including DVC, testing, EDA)
    pip install -r requirements.txt
    ```
    
    **Requirements Files**:
    | File | Purpose |
    | :--- | :--- |
    | `requirements.txt` | Full development dependencies (DVC, Great Expectations, PyTorch with GPU, testing, visualization) |
    | `requirements-prod.txt` | Minimal production dependencies for Docker (Flask, PyTorch CPU-only, MLflow) |

4.  **Configure Kaggle Credentials**
    The dataset is downloaded from Kaggle. Set up credentials:
    ```bash
    # Create .env file with:
    KAGGLE_USERNAME=your_username
    KAGGLE_KEY=your_api_key
    ```

5.  **Run End-to-End Pipeline with DVC**
    The project uses **DVC (Data Version Control)** for reproducible ML pipelines. Execute the full pipeline:
    ```bash
    # Run the complete pipeline
    dvc repro
    
    # Or run individual stages
    dvc repro data_ingestion
    dvc repro data_validation
    dvc repro data_preparation
    dvc repro data_transformation
    dvc repro model_training
    ```
    
    **Useful DVC Commands**:
    ```bash
    # View pipeline DAG
    dvc dag
    
    # Check pipeline status
    dvc status
    
    # Force re-run all stages
    dvc repro --force
    ```

    **DVC Pipeline Stages** (defined in `dvc.yaml`):
    | Stage | Script | Outputs |
    | :--- | :--- | :--- |
    | `data_ingestion` | `src/data_ingestion/ingestion.py` | `data/raw/` |
    | `data_validation` | `src/data_validation/validation.py` | `reports/validation/` |
    | `data_preparation` | `src/data_preparation/preparation.py` | `data/prepared/`, `artifacts/eda/` |
    | `data_transformation` | `src/data_transformation/transformation.py` | `data/transformed/` |
    | `model_training` | `src/model_building/train_model.py` | `models/`, `mlflow.db` |

    **Pipeline Flow**:
    1.  **Data Ingestion**:
        -   Downloads the Cats & Dogs dataset from Kaggle (`bhavikjikadara/dog-and-cat-classification-dataset`).
        -   Saves raw images to `data/raw/PetImages`.
    2.  **Data Validation**:
        -   Uses **Great Expectations** to validate the downloaded images.
        -   Checks for valid extensions (`.jpg`, `.jpeg`, `.png`), expected classes (`cat`, `dog`), and minimum samples per class.
        -   Generates validation reports in `reports/validation`.
    3.  **Data Preparation**:
        -   Cleans corrupt/invalid images.
        -   Generates EDA reports (sample distributions, class balance).
        -   **Output Location**: EDA artifacts saved to `artifacts/eda/`.
    4.  **Data Transformation**:
        -   Resizes all images to 224x224 pixels.
        -   Splits data into train (80%), validation (10%), and test (10%) sets.
        -   Saves transformed images to `data/transformed/train`, `data/transformed/val`, `data/transformed/test`.
    5.  **Model Training**:
        -   Uses **Transfer Learning with ResNet18** (pretrained on ImageNet).
        -   Trains with Adam optimizer, Cross-Entropy Loss, and AMP (Automatic Mixed Precision).
        -   Logs all params, metrics, and artifacts to **MLflow**.
        -   Registers the best model as `CatsDogsCNN` in MLflow Model Registry.
    
    **Parameters**: All pipeline parameters are centralized in `params.yaml` for easy configuration.

6.  **Start API Server**
    ```bash
    python src/model_serving/app.py
    ```

7.  **Validate Deployment**
    -   **Health Check**:
        ```bash
        curl http://localhost:5000/health
        ```
    -   **CLI Test (Single Request)**:
        ```bash
        curl -X POST -F "file=@path/to/cat_or_dog_image.jpg" http://localhost:5000/predict
        ```
    -   **UI Test**: Open a browser and navigate to:
        ```
        http://localhost:5000/
        ```
        This loads a modern web UI where you can upload an image and get real-time predictions with confidence scores.

## 2. EDA and Modelling Choices

### Dataset
-   **Source**: Kaggle - "Dog and Cat Classification Dataset"
-   **Classes**: `cat`, `dog` (Binary Classification)
-   **Total Images**: ~25,000 images (balanced between classes)

### Exploratory Data Analysis (EDA)
The data preparation pipeline generates analysis reports to validate data quality:

-   **Artifact Location**: `artifacts/eda/`
-   **Generated Insights**:
    1.  **Class Distribution**: Verified balanced dataset between cats and dogs.
    2.  **Image Validation**: Identified and removed corrupt/invalid images.
    3.  **Sample Visualization**: Preview of sample images from each class.

### Preprocessing Pipeline
Images are transformed consistently for model training:
1.  **Resize**: All images scaled to 224x224 pixels.
2.  **Normalization**: ImageNet statistics applied:
    -   Mean: `[0.485, 0.456, 0.406]`
    -   Std: `[0.229, 0.224, 0.225]`
3.  **Data Splits**: 80% train, 10% validation, 10% test (stratified).

### Modelling Methodology
We use **Transfer Learning** with a pretrained ResNet18 architecture for efficient and accurate classification.

#### Model Architecture
| Component | Configuration |
| :--- | :--- |
| **Base Model** | ResNet18 (pretrained on ImageNet) |
| **Final Layer** | Linear (512 → 2 classes) |
| **Optimizer** | Adam (LR: 1e-4) |
| **Loss Function** | Cross-Entropy |
| **Training Device** | CUDA GPU (with CPU fallback) |

#### Training Configuration
| Parameter | Value |
| :--- | :--- |
| Image Size | 224x224 |
| Batch Size | 32 |
| Epochs | 5 (configurable in `params.yaml`) |
| Learning Rate | 0.0001 |
| Mixed Precision | Enabled (AMP) |

#### Model Performance Metrics
The model is evaluated on the test set with the following metrics:

| Metric | Description |
| :--- | :--- |
| **Accuracy** | Overall correct predictions |
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under the ROC curve |

#### Selection Logic
The automated selection logic prioritizes **F1 Score** as the primary metric. The best performing model during training is automatically saved and registered in MLflow.

## 3. Experiment Tracking Summary

We integrated **MLflow** to track the entire machine learning lifecycle, ensuring reproducibility and observability.

### 3.1 Configuration & Setup
-   **Backend Store**: SQLite (`sqlite:///mlflow.db`) for lightweight local metadata storage.
-   **Artifact Store**: Local directory `mlruns/` for storing models and training artifacts.
-   **Experiment Name**: `Cats vs Dogs CNN`
-   **Run Name**: `ResNet18_GPU`

### 3.2 Tracked Metrics & Parameters
For each training run, the following are logged:
-   **Hyperparameters**:
    -   `architecture`: ResNet18
    -   `epochs`: Number of training epochs
    -   `batch_size`: 32
    -   `learning_rate`: 0.0001
    -   `image_size`: 224
    -   `device`: cuda/cpu
-   **Performance Metrics** (per epoch):
    -   `train_loss`: Training loss
    -   `val_accuracy`, `val_precision`, `val_recall`, `val_f1`, `val_roc_auc`
-   **Test Metrics** (final):
    -   `test_accuracy`, `test_precision`, `test_recall`, `test_f1`, `test_roc_auc`

### 3.3 Model Registry & Artifacts
-   **Artifacts Preserved**:
    -   `model/`: The serialized PyTorch model (MLflow format).
    -   `MLmodel`: Metadata defining the model flavor and dependencies.
    -   `conda.yaml` / `requirements.txt`: Environment definitions for reproduction.
-   **Registration Strategy**:
    The training script automatically registers the best model in the MLflow Model Registry as:
    -   **Model Name**: `CatsDogsCNN`
    -   **Stage**: Ready for Production / Deployment.

### 3.4 Model Export
-   The best model is also exported to `models/model_export/` in MLflow format for Docker containerization.
-   A performance report is generated at `src/model_building/model_performance_report.md`.

## 4. Architecture Diagram

The system follows a microservices architecture pattern with offline training and online serving.

```mermaid
graph TD
    subgraph "Data Pipeline Orchestrator"
        Ingest[Data Ingestion<br/>(Kaggle Download)] --> Validate[Data Validation<br/>(Great Expectations)]
        Validate --> Prep[Data Preparation<br/>(EDA & Cleaning)]
        Prep --> Transform[Data Transformation<br/>(Resize & Split)]
        Transform --> Train[Model Training<br/>(ResNet18 CNN)]
    end

    subgraph "MLOps Infrastructure"
        Train -->|Log Metrics & Model| MLflow[MLflow Tracking<br/>(SQLite)]
        MLflow -->|Register Best Model| Registry[Model Registry<br/>(CatsDogsCNN)]
    end

    subgraph "Production Deployment"
        Registry -->|Load Model| App[Flask API<br/>(app.py)]
        App -->|Containerize| Docker[Docker Image]
        Docker -->|Deploy| K8s[Kubernetes Cluster]
        K8s -->|Expose| API[REST API :5000]
    end

    subgraph "Monitoring"
        App -->|Metrics| Prometheus[Prometheus<br/>(prometheus-flask-exporter)]
        Prometheus --> Grafana[Grafana Dashboard]
    end

    User[End User] -->|Upload Image| API
    API -->|Cat/Dog Prediction| User
```

## 5. CI/CD and Deployment Workflow

### CI/CD Pipeline
Managed via **GitHub Actions** (`.github/workflows/ci_cd_pipeline.yml`).
-   **Trigger**: Push or Pull Request to branches `main` or `master`.
-   **Jobs**:

    #### 1. Build, Test, and Train (`build-test-train`)
    -   **Environment**: Ubuntu runner with Python 3.9.
    -   **Linting**: Uses `flake8` to enforce PEP8 standards.
    -   **Unit Testing**: Executes `pytest` on the `tests/` directory.
    -   **Pipeline Execution**: Runs the full data pipeline including model training.
    -   **Artifact Archival**:
        -   `mlflow-runs` and `mlflow.db`: Trained model and tracking database.
        -   `model-export`: Exported model for Docker.
        -   `model-performance-report`: Markdown summary of metrics.
        -   `validation-reports` and `pipeline-logs`.

    #### 2. Build and Deploy (`build-and-deploy`)
    -   **Condition**: Runs only on `push` events.
    -   **Docker Build**: Uses Docker Buildx to build the image from the `Dockerfile`.
    -   **Kind Cluster**: Creates a local Kubernetes cluster using Kind.
    -   **Deployment**:
        -   Loads the Docker image into Kind.
        -   Applies `k8/deployment.yaml` and `k8/service.yaml`.
    -   **Smoke Tests**: Runs automated tests against the deployed service.
    -   **Verification**: Validates rollout with `kubectl rollout status`.

### Deployment Configuration

-   **Containerization** (`Dockerfile`):
    -   **Base Image**: `python:3.10-slim`.
    -   **Model Bundling**: Copies `models/model_export` to `/app/model`.
    -   **Environment**: Sets `MLFLOW_TRACKING_URI` for model loading.
    -   **Exposed Port**: 5000.
    -   **Entrypoint**: Starts the Flask application.

-   **Orchestration** (`k8/` directory):
    -   **Deployment** (`deployment.yaml`):
        -   **Replicas**: 2 for high availability.
        -   **Resources**: CPU/Memory limits configured.
        -   **Health Probes**: Readiness and liveness probes on `/health`.
    -   **Service** (`service.yaml`): Exposes the application via LoadBalancer on port 5000.
    -   **Monitoring** (`service-monitor.yaml`): Configures Prometheus scraping for metrics.

### Flask API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | GET | Web UI for image upload and prediction |
| `/predict` | POST | Accepts image file, returns prediction and confidence |
| `/health` | GET | Health check endpoint for Kubernetes probes |
| `/metrics` | GET | Prometheus metrics endpoint |

### Monitoring Stack
-   **Prometheus**: Scrapes application metrics via `prometheus-flask-exporter`.
-   **Grafana**: Visualizes metrics dashboards.
-   **Loki + Promtail**: Log aggregation and viewing.

## 6. Link to Code Repository

https://github.com/Ezaz13/MLOPS_Cats_Dogs_classification.git

## 7. Link to Application Walkthrough Recording
https://wilpbitspilaniacin0-my.sharepoint.com/:v:/g/personal/2024aa05083_wilp_bits-pilani_ac_in/IQBYKzr_iTrITo2C0iiXpQMpAQFmPtSfN-VtvlFs1QJn9Vg?e=RAE8qm&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D