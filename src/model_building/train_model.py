"""
End-to-End CNN Training, Evaluation & Experiment Tracking
Cats vs Dogs Classification (Computer Vision, MLOps)

Features:
- Transfer Learning (ResNet18)
- Explicit CPU / GPU logging
- Batch-level training logs
- AMP (mixed precision)
- MLflow experiment tracking (SQLite)
- Windows-safe multiprocessing
"""

import os
import sys
import time
import logging
import datetime
import json
import numpy as np
import torch
import mlflow
import mlflow.pytorch

from pathlib import Path
from torch import nn, optim, amp
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ------------------------------------------------------------------
# PROJECT SETUP
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# ------------------------------------------------------------------
# LOGGING CONFIG
# ------------------------------------------------------------------
logging.getLogger("alembic").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.ERROR)

from src.utility.logger import setup_logging
logger = setup_logging("model_training")

# ------------------------------------------------------------------
# DATA PATHS
# ------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "data" / "transformed"

# ------------------------------------------------------------------
# TRAINING CONFIG
# ------------------------------------------------------------------
IS_CI = os.getenv("CI", "false").lower() == "true"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 1 if IS_CI else 1
LR = 1e-4

if IS_CI:
    logger.warning("CI ENVIRONMENT DETECTED — RUNNING IN SMOKE TEST MODE (1 Epoch, Reduced Data)")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 if os.name == "nt" else os.cpu_count() // 2

# ------------------------------------------------------------------
# DEVICE LOGS
# ------------------------------------------------------------------
logger.info("========================================")
logger.info(f"DEVICE IN USE : {DEVICE}")
# Force print to ensure visibility even if logging fails
print(f"Training running on: {DEVICE}")

if DEVICE.type == "cuda":
    logger.info(f"GPU NAME     : {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA VERSION : {torch.version.cuda}")
else:
    logger.warning("GPU NOT AVAILABLE — RUNNING ON CPU")

logger.info("========================================")

# ------------------------------------------------------------------
# DATA LOADERS
# ------------------------------------------------------------------
def get_dataloaders():
    logger.info("Preparing datasets...")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    loaders = {}
    datasets_map = {}

    for split in ["train", "val", "test"]:
        dataset = datasets.ImageFolder(DATA_ROOT / split, transform=transform)

        if split == "train":
            class_names = dataset.classes

        if IS_CI:
            # Use only a small subset for smoke testing
            subset_size = min(len(dataset), 100) # 100 images (~3 batches)
            dataset = Subset(dataset, range(subset_size))

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE.type == "cuda")
        )

        loaders[split] = loader

        logger.info(
            f"{split.upper()} SET -> "
            f"Images: {len(dataset)} | Batches: {len(loader)}"
        )

    return loaders, class_names

# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------
def build_model(num_classes):
    logger.info("Building ResNet18 (pretrained on ImageNet)...")
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)
    logger.info("Model successfully moved to device.")
    return model

# ------------------------------------------------------------------
# TRAIN ONE EPOCH (WITH BATCH LOGS)
# ------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    losses = []
    start_time = time.time()

    logger.info(f"---- STARTING EPOCH {epoch + 1} ----")

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

        # Log every 50 batches
        if batch_idx % 50 == 0:
            logger.info(
                f"[Epoch {epoch+1}] "
                f"Batch {batch_idx}/{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    duration = time.time() - start_time
    logger.info(
        f"---- EPOCH {epoch + 1} COMPLETED "
        f"in {duration/60:.2f} minutes ----"
    )

    return float(np.mean(losses))

# ------------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------------
def evaluate(model, loader, phase):
    logger.info(f"Evaluating on {phase.upper()} set...")
    model.eval()

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        logger.warning(f"ROC AUC undefined (likely single class in {phase} batch). Setting to 0.0")
        metrics["roc_auc"] = 0.0

    logger.info(f"{phase.upper()} METRICS: {metrics}")
    return metrics

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    logger.info("========== TRAINING PIPELINE STARTED ==========")

    # Create models directory for DVC output
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create export directory for Docker/CI
    model_export_dir = models_dir / "model_export"
    model_export_dir.mkdir(parents=True, exist_ok=True)

    report_path = PROJECT_ROOT / "src" / "model_building" / "model_performance_report.md"

    # Use absolute path for MLflow database
    mlflow_db_path = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    
    # Ensure mlruns directory exists (though MLflow should create it)
    mlruns_dir = PROJECT_ROOT / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_name = "Cats vs Dogs CNN"
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=str(mlruns_dir.as_uri()) 
        )
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
    mlflow.set_experiment(experiment_id=experiment_id)
    REGISTERED_MODEL_NAME = "CatsDogsCNN"

    loaders, classes = get_dataloaders()
    model = build_model(num_classes=len(classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_f1 = -1.0
    best_run_id = None
    best_model_state = None

    with mlflow.start_run(run_name="ResNet18_GPU"):
        mlflow.log_params({
            "architecture": "ResNet18",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "image_size": IMAGE_SIZE,
            "device": str(DEVICE),
        })

        for epoch in range(EPOCHS):
            train_loss = train_epoch(
                model, loaders["train"], criterion, optimizer, scaler, epoch
            )
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            val_metrics = evaluate(model, loaders["val"], "val")
            mlflow.log_metrics(
                {f"val_{k}": v for k, v in val_metrics.items()},
                step=epoch
            )

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_run_id = mlflow.active_run().info.run_id
                best_model_state = model.state_dict().copy()

                # Log model WITHOUT input_example (avoids GPU/CPU warnings)
                mlflow.pytorch.log_model(
                    model,
                    name="model"
                )

        test_metrics = evaluate(model, loaders["test"], "test")
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Generate Model Performance Report
        report_content = f"""# Model Performance Report

*Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

*MLflow Experiment: 'Cats vs Dogs CNN'*

## Best Model: ResNet18

- **MLflow Run ID**: `{best_run_id if best_run_id else mlflow.active_run().info.run_id}`
- **Device**: `{DEVICE}`
- **Epochs**: {EPOCHS}
- **Batch Size**: {BATCH_SIZE}
- **Learning Rate**: {LR}
- **Image Size**: {IMAGE_SIZE}

### Metrics (Test Set - Final Epoch)
- **Accuracy**: {test_metrics.get('accuracy', 0.0):.4f}
- **Precision**: {test_metrics.get('precision', 0.0):.4f}
- **Recall**: {test_metrics.get('recall', 0.0):.4f}
- **F1-Score**: {test_metrics.get('f1', 0.0):.4f}
- **ROC-AUC**: {test_metrics.get('roc_auc', 0.0):.4f}
"""
        with open(report_path, "w") as f:
            f.write(report_content)
        logger.info(f"Model performance report saved to {report_path}")

    if best_run_id:
        mlflow.register_model(
            model_uri=f"runs:/{best_run_id}/model",
            name=REGISTERED_MODEL_NAME
        )
        logger.info("Best model registered in MLflow.")
        
        # Save best model to local models directory for DVC and Export
        if best_model_state is not None:
            # Save to main models dir (for DVC if needed)
            model_path_dvc = models_dir / "best_model.pth"
            model_data = {
                'model_state_dict': best_model_state,
                'f1_score': best_f1,
                'run_id': best_run_id,
                'architecture': 'ResNet18',
                'num_classes': len(classes),
            }
            torch.save(model_data, model_path_dvc)
            
            # Save to export dir (for Docker/CI artifact)
            import shutil
            if model_export_dir.exists():
                shutil.rmtree(model_export_dir)
            
            mlflow.pytorch.save_model(model, path=str(model_export_dir))
            
            logger.info(f"Best model saved to {model_path_dvc} and exported to {model_export_dir}")

    logger.info("========== TRAINING PIPELINE COMPLETED ==========")

# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
