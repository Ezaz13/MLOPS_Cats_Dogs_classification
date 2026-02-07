import os
import sys
import zipfile
from pathlib import Path
from tqdm import tqdm

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utility.exception import CustomException
from src.utility.logger import setup_logging

# -------------------------------------------------------------------
# Setup logging
# -------------------------------------------------------------------
logger = setup_logging("ingestion")

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
KAGGLE_DATASET = "bhavikjikadara/dog-and-cat-classification-dataset"


# -------------------------------------------------------------------
# Ingestion Logic
# -------------------------------------------------------------------
def load_env_vars():
    """
    Manually load environment variables from a .env file in the project root.
    This allows using API tokens without hardcoding them in the script.
    """
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        logger.info(f"Loading environment variables from {env_path}")
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        except Exception as e:
            logger.warning(f"Failed to load .env file: {e}")

def download_from_kaggle(dataset: str, dest_path: Path):
    """
    Download and extract dataset from Kaggle.
    Requires kaggle.json to be set up in ~/.kaggle/ or via environment variables.
    """
    try:
        # -------------------------------------------------------------------
        # Kaggle Authentication Setup
        # -------------------------------------------------------------------
        load_env_vars()
        
        # Alternative: Set credentials directly here (not recommended for shared code)
        # os.environ['KAGGLE_USERNAME'] = "your_username"
        # os.environ['KAGGLE_KEY'] = "your_key"

        from kaggle.api.kaggle_api_extended import KaggleApi

        logger.info(f"Authenticating with Kaggle...")
        api = KaggleApi()
        api.authenticate()

        logger.info(f"Downloading dataset: {dataset}")

        dest_path.mkdir(parents=True, exist_ok=True)
        
        # Use the standard API method to download and unzip
        # This replaces the manual streaming which was causing AttributeError
        api.dataset_download_files(dataset, path=dest_path, unzip=True)

        logger.info(f"Dataset downloaded and extracted successfully.")
    except Exception as e:
        error_msg = (
            f"Failed to download from Kaggle: {e}\n"
            "Please check your Kaggle credentials. You can:\n"
            "1. Place kaggle.json in C:\\Users\\<User>\\.kaggle\\\n"
            "2. Create a .env file in the project root with KAGGLE_USERNAME and KAGGLE_KEY\n"
            "3. Set environment variables manually."
        )
        raise CustomException(error_msg, sys)


def ingest_source_data():
    try:
        # Create raw data directory
        RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

        extract_to = RAW_DATA_PATH

        logger.info(f"Starting data ingestion into directory: {extract_to}")
        download_from_kaggle(KAGGLE_DATASET, extract_to)

        logger.info("Data ingestion completed successfully.")
        return True

    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}", exc_info=True)
        return False


# -------------------------------------------------------------------
# Standalone Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Running Cats vs Dogs ingestion script as standalone process.")
    success = ingest_source_data()

    if success:
        sys.exit(0)
    else:
        sys.exit(1)