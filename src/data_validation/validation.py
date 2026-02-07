import sys
import os
from pathlib import Path
import pandas as pd
from PIL import Image

# -------------------------------------------------------------------
# Project setup
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utility.exception import CustomException
from src.utility.logger import setup_logging

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
logger = setup_logging("validation")

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "PetImages"
EXPECTED_CLASSES = ["cat", "dog"]
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# -------------------------------------------------------------------
# Validator
# -------------------------------------------------------------------
class ImageDatasetValidator:

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.stats = {
            "total_files": 0,
            "valid_images": 0,
            "corrupt_images": 0,
            "invalid_format": 0,
            "class_counts": {cls: 0 for cls in EXPECTED_CLASSES},
        }

    # ---------------------------------------------------------------
    def validate_structure(self):
        logger.info("Validating dataset structure...")

        if not self.data_path.exists():
            raise CustomException(
                f"Dataset root not found: {self.data_path}", sys
            )

        for cls in EXPECTED_CLASSES:
            cls_path = self.data_path / cls.capitalize()
            if not cls_path.exists():
                raise CustomException(
                    f"Missing class folder: {cls_path}", sys
                )

        logger.info("Dataset structure is valid.")

    # ---------------------------------------------------------------
    def validate_images(self):
        logger.info("Validating images...")

        for cls in EXPECTED_CLASSES:
            class_dir = self.data_path / cls.capitalize()

            for file in class_dir.iterdir():
                if not file.is_file():
                    continue

                self.stats["total_files"] += 1

                if file.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                    self.stats["invalid_format"] += 1
                    continue

                try:
                    with Image.open(file) as img:
                        img.verify()

                    with Image.open(file) as img:
                        img.load()

                    self.stats["valid_images"] += 1
                    self.stats["class_counts"][cls] += 1

                except Exception as e:
                    logger.warning(f"Corrupt image: {file} | {e}")
                    self.stats["corrupt_images"] += 1

        # Fail if any class is empty
        for cls, count in self.stats["class_counts"].items():
            if count == 0:
                raise CustomException(
                    f"No valid images found for class '{cls}'", sys
                )

        return self.stats["corrupt_images"] == 0

    # ---------------------------------------------------------------
    def generate_report(self):
        logger.info("Generating validation report...")

        report_dir = PROJECT_ROOT / "reports" / "validation"
        report_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([self.stats])
        report_path = report_dir / "image_validation_summary.csv"
        df.to_csv(report_path, index=False)

        logger.info(f"Report saved to: {report_path}")

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        validator = ImageDatasetValidator(RAW_DATA_PATH)

        validator.validate_structure()
        success = validator.validate_images()
        validator.generate_report()

        if success:
            logger.info("Data validation completed successfully.")
            sys.exit(0)
        else:
            logger.warning("Validation completed with corrupt images.")
            sys.exit(0)

    except Exception as e:
        logger.error("Data validation failed.", exc_info=True)
        sys.exit(1)
