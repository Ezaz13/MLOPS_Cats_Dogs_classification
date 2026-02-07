import os
import subprocess
import sys
from pathlib import Path
from src.utility.logger import setup_logging
from src.utility.exception import CustomException

# -------------------------------------------------------------------
# Setup logging
# -------------------------------------------------------------------
logger = setup_logging("dvc_init")

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

def run_command(command, cwd=PROJECT_ROOT):
    """Executes a shell command and logs output."""
    try:
        logger.info(f"Running command: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.stderr}")
        return False

def initialize_dvc():
    """Initializes DVC and configures local storage."""
    try:
        # 1. Initialize DVC if not already initialized
        if not (PROJECT_ROOT / ".dvc").exists():
            logger.info("Initializing DVC...")
            if not run_command(["dvc", "init"]):
                raise Exception("Failed to initialize DVC")
        else:
            logger.info("DVC is already initialized.")

        # 2. Configure DVC remote (Local storage for simulation)
        # In a real scenario, this would be S3, GDrive, etc.
        dvc_storage = PROJECT_ROOT / "dvc_storage"
        dvc_storage.mkdir(exist_ok=True)
        
        logger.info(f"Configuring local DVC remote at {dvc_storage}")
        
        # Add remote 'myremote'
        run_command(["dvc", "remote", "add", "-d", "myremote", str(dvc_storage)])
        
        # 3. Add data to DVC
        if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
            logger.info(f"Adding data directory to DVC: {DATA_DIR}")
            if run_command(["dvc", "add", str(DATA_DIR)]):
                logger.info("Data added to DVC successfully.")
            else:
                logger.warning("Failed to add data to DVC.")
        else:
            logger.warning(f"Data directory {DATA_DIR} is empty or does not exist. Run ingestion first.")

        # 4. Git tracking
        logger.info("Configuring Git tracking for DVC files...")
        run_command(["git", "add", ".dvc", "dvc.yaml", "dvc.lock", "data/raw.dvc", ".gitignore"])
        # Note: We don't commit automatically to give user control, but we stage the files.

        logger.info("DVC initialization and configuration complete.")
        
    except Exception as e:
        raise CustomException(f"DVC Initialization failed: {e}", sys)

if __name__ == "__main__":
    initialize_dvc()
