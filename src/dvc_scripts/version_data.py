import sys
import subprocess
import argparse
from pathlib import Path
from src.utility.logger import setup_logging
from src.utility.exception import CustomException

# -------------------------------------------------------------------
# Setup logging
# -------------------------------------------------------------------
logger = setup_logging("dvc_versioning")

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

def version_data(message):
    """Adds data to DVC, commits changes to Git, and pushes to DVC remote."""
    try:
        # 1. Add data to DVC (tracks changes)
        logger.info(f"Tracking changes in {DATA_DIR} with DVC...")
        if not run_command(["dvc", "add", str(DATA_DIR)]):
             raise Exception("Failed to run 'dvc add'")

        # 2. Git commit the DVC metadata files
        logger.info("Staging DVC metadata files to Git...")
        # data/raw.dvc is the file created by dvc add
        dvc_file = "data/raw.dvc" 
        run_command(["git", "add", dvc_file, ".gitignore"])
        
        logger.info(f"Committing to Git with message: '{message}'")
        if not run_command(["git", "commit", "-m", message]):
             logger.warning("Git commit failed (maybe nothing to commit?)")

        # 3. Push data to DVC remote
        logger.info("Pushing data to DVC remote storage...")
        if not run_command(["dvc", "push"]):
             raise Exception("Failed to push data to DVC remote")

        logger.info("Data versioning completed successfully.")

    except Exception as e:
        raise CustomException(f"Data versioning failed: {e}", sys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Version control data using DVC and Git")
    parser.add_argument("--message", "-m", type=str, required=True, help="Commit message for the data version")
    args = parser.parse_args()

    version_data(args.message)
