"""
DVC Initialization Script
Automates DVC repository setup with Git integration and remote storage configuration
"""

import os
import sys
import subprocess
from pathlib import Path

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utility.logger import setup_logging

logger = setup_logging("dvc_init")

# Configuration
DVC_REMOTE_NAME = "local_storage"
DVC_REMOTE_PATH = PROJECT_ROOT / "dvc_storage"


def run_command(cmd: list, check=True, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=check,
            capture_output=capture_output,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(cmd)}")
        logger.error(f"Error: {e.stderr}")
        raise


def check_git_initialized():
    """Check if Git repository is initialized."""
    git_dir = PROJECT_ROOT / ".git"
    return git_dir.exists()


def init_git():
    """Initialize Git repository if not already initialized."""
    if check_git_initialized():
        logger.info("Git repository already initialized.")
        return True
    
    logger.info("Initializing Git repository...")
    try:
        run_command(["git", "init"])
        logger.info("Git repository initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Git: {e}")
        return False


def check_dvc_initialized():
    """Check if DVC is initialized."""
    dvc_dir = PROJECT_ROOT / ".dvc"
    return dvc_dir.exists()


def init_dvc():
    """Initialize DVC repository."""
    if check_dvc_initialized():
        logger.info("DVC repository already initialized.")
        return True
    
    logger.info("Initializing DVC repository...")
    try:
        run_command(["dvc", "init"])
        logger.info("DVC repository initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize DVC: {e}")
        return False


def setup_remote_storage():
    """Configure DVC remote storage."""
    logger.info(f"Setting up DVC remote storage: {DVC_REMOTE_NAME}")
    
    # Create remote storage directory
    DVC_REMOTE_PATH.mkdir(parents=True, exist_ok=True)
    logger.info(f"Remote storage directory created: {DVC_REMOTE_PATH}")
    
    try:
        # Check if remote already exists
        result = run_command(["dvc", "remote", "list"], check=False)
        if DVC_REMOTE_NAME in result.stdout:
            logger.info(f"Remote '{DVC_REMOTE_NAME}' already configured.")
            # Update the path in case it changed
            run_command([
                "dvc", "remote", "modify", DVC_REMOTE_NAME, "url", str(DVC_REMOTE_PATH)
            ])
            logger.info(f"Updated remote path to: {DVC_REMOTE_PATH}")
        else:
            # Add new remote
            run_command([
                "dvc", "remote", "add", "-d", DVC_REMOTE_NAME, str(DVC_REMOTE_PATH)
            ])
            logger.info(f"Remote '{DVC_REMOTE_NAME}' added successfully.")
        
        return True
    except Exception as e:
        logger.error(f"Failed to setup remote storage: {e}")
        return False


def configure_dvc_cache():
    """Configure DVC cache settings for better performance and Windows compatibility."""
    logger.info("Configuring DVC cache settings...")
    
    try:
        # Set cache types with fallback options (copy is most compatible)
        # This fixes "No possible cache link types" error on Windows
        run_command(["dvc", "config", "cache.type", "copy,reflink,hardlink"])
        logger.info("DVC cache configured with multiple link types (copy,reflink,hardlink).")
        
        # Disable protected mode to avoid permission issues
        run_command(["dvc", "config", "cache.protected", "false"])
        logger.info("DVC cache protected mode disabled.")
        
        return True
    except Exception as e:
        logger.warning(f"Could not configure cache settings: {e}")
        logger.info("Continuing with default cache settings.")
        return True


def verify_dvc_setup():
    """Verify DVC setup is correct."""
    logger.info("Verifying DVC setup...")
    
    try:
        # Check DVC status
        result = run_command(["dvc", "status"], check=False)
        
        # Check remote configuration
        result = run_command(["dvc", "remote", "list"])
        logger.info(f"Configured remotes:\n{result.stdout}")
        
        logger.info("DVC setup verification completed successfully.")
        return True
    except Exception as e:
        logger.error(f"DVC setup verification failed: {e}")
        return False


def main():
    """Main initialization workflow."""
    logger.info("=" * 60)
    logger.info("DVC INITIALIZATION SCRIPT")
    logger.info("=" * 60)
    
    # Step 1: Initialize Git
    if not init_git():
        logger.error("Git initialization failed. Aborting.")
        sys.exit(1)
    
    # Step 2: Initialize DVC
    if not init_dvc():
        logger.error("DVC initialization failed. Aborting.")
        sys.exit(1)
    
    # Step 3: Setup remote storage
    if not setup_remote_storage():
        logger.error("Remote storage setup failed. Aborting.")
        sys.exit(1)
    
    # Step 4: Configure cache
    configure_dvc_cache()
    
    # Step 5: Verify setup
    if not verify_dvc_setup():
        logger.warning("DVC verification had issues. Please check manually.")
    
    logger.info("=" * 60)
    logger.info("DVC INITIALIZATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Add data to track: dvc add data/raw")
    logger.info("2. Run pipeline: dvc repro")
    logger.info("3. Push data to remote: dvc push")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        sys.exit(1)
