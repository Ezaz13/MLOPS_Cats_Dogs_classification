import sys
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.utility.exception import CustomException
from src.utility.logger import setup_logging

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = setup_logging("data_transformation")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "PetImages"
TRANSFORMED_ROOT = PROJECT_ROOT / "data" / "transformed"

IMAGE_SIZE = (224, 224)
RANDOM_STATE = 42
CLASSES = ["cat", "dog"]

# --------------------------------------------------
# CREATE OUTPUT FOLDERS
# --------------------------------------------------
def create_dirs():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (TRANSFORMED_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD IMAGE PATHS
# --------------------------------------------------
def collect_image_paths():
    records = []

    for cls in CLASSES:
        class_dir = RAW_DATA_PATH / cls.capitalize()
        if not class_dir.exists():
            raise CustomException(f"Missing folder: {class_dir}", sys)

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append((img_path, cls))

    if not records:
        raise CustomException("No images found.", sys)

    return records

# --------------------------------------------------
# SPLIT DATA
# --------------------------------------------------
def split_data(records):
    paths, labels = zip(*records)

    train_p, temp_p, train_l, temp_l = train_test_split(
        paths, labels,
        test_size=0.2,
        stratify=labels,
        random_state=RANDOM_STATE
    )

    val_p, test_p, val_l, test_l = train_test_split(
        temp_p, temp_l,
        test_size=0.5,
        stratify=temp_l,
        random_state=RANDOM_STATE
    )

    return {
        "train": list(zip(train_p, train_l)),
        "val": list(zip(val_p, val_l)),
        "test": list(zip(test_p, test_l)),
    }

# --------------------------------------------------
# TRANSFORM + SAVE
# --------------------------------------------------
def process_split(split_name, samples):
    logger.info(f"Processing {split_name} set...")

    for src_path, label in tqdm(samples, desc=split_name.upper()):
        dst_path = TRANSFORMED_ROOT / split_name / label / src_path.name

        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                img.save(dst_path, format="JPEG", quality=95)

        except Exception as e:
            logger.warning(f"Skipped image {src_path}: {e}")

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    logger.info("Starting Kaggle-style data transformation")

    create_dirs()
    records = collect_image_paths()
    splits = split_data(records)

    for split_name, samples in splits.items():
        process_split(split_name, samples)

    logger.info("Data transformation completed successfully.")
    print("\nDATA TRANSFORMATION COMPLETED SUCCESSFULLY")

# --------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Pipeline failed.", exc_info=True)
        sys.exit(1)
