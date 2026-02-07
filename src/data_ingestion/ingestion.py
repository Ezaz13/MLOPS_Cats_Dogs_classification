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
def download_from_kaggle(dataset: str, dest_path: Path):
    """
    Download and extract dataset from Kaggle.
    Requires kaggle.json to be set up in ~/.kaggle/ or via environment variables.
    """
    try:
        
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        logger.info(f"Authenticating with Kaggle...")
        api = KaggleApi()
        api.authenticate()
        
        logger.info(f"Downloading dataset: {dataset}")
        
        owner, dataset_name = dataset.split('/')
        dest_path.mkdir(parents=True, exist_ok=True)
        zip_path = dest_path / "archive.zip"
        
        # Get stream - _preload_content=False gives us the raw HTTP response to stream
        response = api.datasets_download_by_dataset_handle(
            owner_slug=owner, dataset_slug=dataset_name, _preload_content=False
        )
        
        total_size = int(response.headers.get('Content-Length', 0))
        
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc="Downloading") as pbar:
            with open(zip_path, 'wb') as f:
                while True:
                    chunk = response.read(32 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Extracting to {dest_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
            
        if zip_path.exists():
            os.remove(zip_path)
        
        logger.info(f"Dataset downloaded and extracted successfully.")
    except Exception as e:
        error_msg = (
            f"Failed to download from Kaggle: {e}\n"
            "Please check your Kaggle credentials (username and key) in the script."
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
