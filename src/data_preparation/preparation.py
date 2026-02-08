import os
import sys
import glob
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFile

# ------------------------------------------------------------------
# PIL CONFIG (safe handling of truncated images)
# ------------------------------------------------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------------------------------------------------------------------
# PROJECT SETUP
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utility.exception import CustomException
from src.utility.logger import setup_logging

logger = setup_logging("data_preparation")

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "PetImages"
PREPARED_DATA_PATH = PROJECT_ROOT / "data" / "prepared"
EDA_PATH = PROJECT_ROOT / "artifacts" / "eda"

EXPECTED_CLASSES = ["cat", "dog"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

PREPARED_DATA_PATH.mkdir(parents=True, exist_ok=True)
EDA_PATH.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    logger.info("Loading dataset from raw folder...")

    records = []

    for cls in EXPECTED_CLASSES:
        class_dir = RAW_DATA_PATH / cls.capitalize()

        if not class_dir.exists():
            raise CustomException(f"Missing class folder: {class_dir}", sys)

        for file in class_dir.iterdir():
            if file.suffix.lower() in VALID_EXTENSIONS:
                records.append({
                    "filepath": str(file),
                    "label": cls
                })

    if not records:
        raise CustomException("No images found in dataset.", sys)

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} image paths.")
    return df

# ------------------------------------------------------------------
# CLEAN DATA
# ------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Validating images...")

    valid_rows = []
    widths, heights = [], []

    for _, row in df.iterrows():
        try:
            with Image.open(row["filepath"]) as img:
                img.verify()

            with Image.open(row["filepath"]) as img:
                widths.append(img.width)
                heights.append(img.height)
                valid_rows.append(row)

        except Exception as e:
            logger.warning(f"Corrupt image removed: {row['filepath']} | {e}")

    df_clean = pd.DataFrame(valid_rows)
    df_clean["width"] = widths
    df_clean["height"] = heights

    if df_clean.empty:
        raise CustomException("All images are corrupt.", sys)

    logger.info(f"Removed {len(df) - len(df_clean)} corrupt images.")
    return df_clean

# ------------------------------------------------------------------
# VALIDATE CLASS BALANCE
# ------------------------------------------------------------------
def validate_class_distribution(df: pd.DataFrame):
    counts = df["label"].value_counts()

    logger.info(f"Class distribution: {counts.to_dict()}")

    for cls in EXPECTED_CLASSES:
        if cls not in counts or counts[cls] == 0:
            raise CustomException(f"No samples found for class '{cls}'", sys)

    imbalance_ratio = counts.max() / counts.min()
    if imbalance_ratio > 2:
        logger.warning(f"Class imbalance detected: {counts.to_dict()}")

# ------------------------------------------------------------------
# EDA
# ------------------------------------------------------------------
def perform_eda(df: pd.DataFrame, eda_run_path: Path):
    logger.info("Performing EDA...")
    sns.set_theme(style="whitegrid")

    # Class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="label")
    plt.title("Class Distribution")
    plt.savefig(eda_run_path / "class_distribution.png")
    plt.close()

    # Image dimensions
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df, x="width", y="height",
        hue="label", alpha=0.5
    )
    plt.title("Image Dimensions")
    plt.savefig(eda_run_path / "image_dimensions.png")
    plt.close()

    # Sample images
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    samples = (
        df.groupby("label", group_keys=False)
          .apply(lambda x: x.sample(min(4, len(x)), random_state=42))
    )

    for ax, (_, row) in zip(axes.flatten(), samples.iterrows()):
        try:
            img = Image.open(row["filepath"])
            ax.imshow(img)
            ax.set_title(row["label"])
            ax.axis("off")
        except Exception:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(eda_run_path / "sample_images.png")
    plt.close()

    logger.info(f"EDA saved to {eda_run_path}")

# ------------------------------------------------------------------
# SAVE PREPARED DATA
# ------------------------------------------------------------------
def save_prepared_data(df: pd.DataFrame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = PREPARED_DATA_PATH / f"prepared_data_{timestamp}.csv"

    df[["filepath", "label", "width", "height"]].to_csv(output_file, index=False)
    df.to_csv(PREPARED_DATA_PATH / "prepared_data_latest.csv", index=False)

    logger.info(f"Prepared dataset saved to {output_file}")

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    logger.info("Starting Data Preparation Pipeline")

    df_raw = load_dataset()
    df_clean = clean_data(df_raw)
    validate_class_distribution(df_clean)

    eda_run_path = EDA_PATH / f"eda_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    eda_run_path.mkdir(parents=True, exist_ok=True)
    perform_eda(df_clean, eda_run_path)

    save_prepared_data(df_clean)

    logger.info("Data Preparation Pipeline completed successfully.")
    print("\nDATA PREPARATION PIPELINE COMPLETED SUCCESSFULLY")

# ------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Pipeline failed.", exc_info=True)
        sys.exit(1)
