"""
DVC Data Versioning Script
Handles data tracking, version tagging, and remote storage operations
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utility.logger import setup_logging

logger = setup_logging("dvc_versioning")


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


def track_directory(directory: str):
    """Track a directory with DVC."""
    dir_path = PROJECT_ROOT / directory
    
    if not dir_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return False
    
    logger.info(f"Tracking directory with DVC: {directory}")
    
    try:
        # Add directory to DVC
        run_command(["dvc", "add", directory])
        logger.info(f"Successfully tracked: {directory}")
        
        # Add .dvc file to Git
        dvc_file = f"{directory}.dvc"
        if Path(PROJECT_ROOT / dvc_file).exists():
            run_command(["git", "add", dvc_file, f"{directory}.gitignore"])
            logger.info(f"Added {dvc_file} to Git")
        
        return True
    except Exception as e:
        logger.error(f"Failed to track {directory}: {e}")
        return False


def push_to_remote():
    """Push tracked data to DVC remote storage."""
    logger.info("Pushing data to DVC remote storage...")
    
    try:
        result = run_command(["dvc", "push"])
        logger.info("Data pushed successfully to remote storage.")
        return True
    except Exception as e:
        logger.error(f"Failed to push data: {e}")
        return False


def pull_from_remote():
    """Pull tracked data from DVC remote storage."""
    logger.info("Pulling data from DVC remote storage...")
    
    try:
        result = run_command(["dvc", "pull"])
        logger.info("Data pulled successfully from remote storage.")
        return True
    except Exception as e:
        logger.error(f"Failed to pull data: {e}")
        return False


def create_version_tag(tag_name: str = None, message: str = None):
    """Create a Git tag for the current data version."""
    if tag_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag_name = f"data_v{timestamp}"
    
    if message is None:
        message = f"Data version created at {datetime.now()}"
    
    logger.info(f"Creating version tag: {tag_name}")
    
    try:
        # Commit DVC files
        run_command(["git", "add", "*.dvc", ".gitignore"], check=False)
        run_command(["git", "commit", "-m", f"Update data version: {tag_name}"], check=False)
        
        # Create tag
        run_command(["git", "tag", "-a", tag_name, "-m", message])
        logger.info(f"Version tag created: {tag_name}")
        
        return tag_name
    except Exception as e:
        logger.error(f"Failed to create version tag: {e}")
        return None


def list_versions():
    """List all data versions (Git tags)."""
    logger.info("Listing data versions...")
    
    try:
        result = run_command(["git", "tag", "-l", "data_v*"])
        versions = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        if versions:
            logger.info(f"Found {len(versions)} data versions:")
            for version in versions:
                logger.info(f"  - {version}")
        else:
            logger.info("No data versions found.")
        
        return versions
    except Exception as e:
        logger.error(f"Failed to list versions: {e}")
        return []


def checkout_version(tag_name: str):
    """Checkout a specific data version."""
    logger.info(f"Checking out data version: {tag_name}")
    
    try:
        # Checkout Git tag
        run_command(["git", "checkout", tag_name])
        
        # Checkout DVC data
        run_command(["dvc", "checkout"])
        
        logger.info(f"Successfully checked out version: {tag_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to checkout version: {e}")
        return False


def generate_version_report():
    """Generate a report of current data versions and status."""
    logger.info("Generating data version report...")
    
    report_dir = PROJECT_ROOT / "reports" / "versioning"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"version_report_{timestamp}.txt"
    
    try:
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("DATA VERSION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            # DVC status
            f.write("DVC STATUS:\n")
            f.write("-" * 70 + "\n")
            result = run_command(["dvc", "status"], check=False)
            f.write(result.stdout if result.stdout else "All pipelines are up to date.\n")
            f.write("\n")
            
            # Tracked data
            f.write("TRACKED DATA:\n")
            f.write("-" * 70 + "\n")
            result = run_command(["dvc", "list", ".", "-R"], check=False)
            f.write(result.stdout if result.stdout else "No data tracked.\n")
            f.write("\n")
            
            # Data versions
            f.write("DATA VERSIONS:\n")
            f.write("-" * 70 + "\n")
            versions = list_versions()
            if versions:
                for version in versions:
                    f.write(f"  {version}\n")
            else:
                f.write("No versions found.\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        logger.info(f"Report generated: {report_file}")
        return report_file
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        return None


def main():
    """Main versioning workflow."""
    logger.info("=" * 60)
    logger.info("DVC DATA VERSIONING SCRIPT")
    logger.info("=" * 60)
    
    # Common directories to track
    directories_to_track = [
        "data/raw",
        "data/prepared",
        "data/transformed",
    ]
    
    logger.info("Tracking data directories...")
    for directory in directories_to_track:
        track_directory(directory)
    
    # Push to remote
    logger.info("\nPushing data to remote storage...")
    push_to_remote()
    
    # Generate report
    logger.info("\nGenerating version report...")
    generate_version_report()
    
    logger.info("=" * 60)
    logger.info("DATA VERSIONING COMPLETED")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Commit changes to Git: git commit -m 'Update data version'")
    logger.info("2. Create version tag: git tag -a data_v1.0 -m 'Version 1.0'")
    logger.info("3. Push to Git: git push && git push --tags")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Versioning failed: {e}", exc_info=True)
        sys.exit(1)
