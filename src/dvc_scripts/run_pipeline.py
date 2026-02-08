"""
DVC Pipeline Runner Script
Orchestrates DVC pipeline execution with enhanced logging and error handling
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utility.logger import setup_logging

logger = setup_logging("dvc_pipeline")


def run_command(cmd: list, check=True, capture_output=False):
    """Run a shell command and return the result."""
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
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
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        raise


def show_pipeline_dag():
    """Display the pipeline DAG (Directed Acyclic Graph)."""
    logger.info("Displaying pipeline DAG...")
    print("\n" + "=" * 70)
    print("PIPELINE DAG")
    print("=" * 70)
    
    try:
        result = run_command(["dvc", "dag"], capture_output=True)
        print(result.stdout)
        return True
    except Exception as e:
        logger.error(f"Failed to generate DAG: {e}")
        return False


def show_pipeline_status():
    """Show the current status of the pipeline."""
    logger.info("Checking pipeline status...")
    print("\n" + "=" * 70)
    print("PIPELINE STATUS")
    print("=" * 70)
    
    try:
        result = run_command(["dvc", "status"], capture_output=True)
        if result.stdout.strip():
            print(result.stdout)
        else:
            print("✓ All pipelines are up to date.")
        return True
    except Exception as e:
        logger.error(f"Failed to check status: {e}")
        return False


def show_pipeline_metrics():
    """Display pipeline metrics."""
    logger.info("Displaying pipeline metrics...")
    print("\n" + "=" * 70)
    print("PIPELINE METRICS")
    print("=" * 70)
    
    try:
        result = run_command(["dvc", "metrics", "show"], capture_output=True, check=False)
        if result.stdout.strip():
            print(result.stdout)
        else:
            print("No metrics available yet.")
        return True
    except Exception as e:
        logger.warning(f"Metrics not available: {e}")
        return False


def run_full_pipeline(force=False):
    """Run the complete DVC pipeline."""
    logger.info("=" * 70)
    logger.info("RUNNING FULL DVC PIPELINE")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    try:
        cmd = ["dvc", "repro"]
        if force:
            cmd.append("--force")
            logger.info("Running with --force flag (will re-run all stages)")
        
        # Run pipeline
        result = run_command(cmd, capture_output=False)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 70)
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY in {duration}")
        logger.info("=" * 70)
        
        return True
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return False


def run_single_stage(stage_name: str, force=False):
    """Run a single pipeline stage."""
    logger.info(f"Running pipeline stage: {stage_name}")
    
    try:
        cmd = ["dvc", "repro", stage_name]
        if force:
            cmd.append("--force")
        
        result = run_command(cmd, capture_output=False)
        logger.info(f"Stage '{stage_name}' completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Stage execution failed: {e}")
        return False


def run_pipeline_dry_run():
    """Run a dry-run to see what would be executed."""
    logger.info("Running pipeline dry-run...")
    print("\n" + "=" * 70)
    print("PIPELINE DRY-RUN")
    print("=" * 70)
    
    try:
        result = run_command(["dvc", "repro", "--dry"], capture_output=True)
        print(result.stdout if result.stdout else "Nothing to run.")
        return True
    except Exception as e:
        logger.error(f"Dry-run failed: {e}")
        return False


def show_params():
    """Display pipeline parameters."""
    logger.info("Displaying pipeline parameters...")
    print("\n" + "=" * 70)
    print("PIPELINE PARAMETERS")
    print("=" * 70)
    
    try:
        result = run_command(["dvc", "params", "diff"], capture_output=True, check=False)
        if result.stdout.strip():
            print(result.stdout)
        else:
            print("No parameter changes detected.")
        return True
    except Exception as e:
        logger.warning(f"Parameters not available: {e}")
        return False


def main():
    """Main pipeline orchestration."""
    parser = argparse.ArgumentParser(
        description="DVC Pipeline Runner for Cats & Dogs Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show pipeline structure
  python run_pipeline.py --dag
  
  # Check pipeline status
  python run_pipeline.py --status
  
  # Dry-run (show what would execute)
  python run_pipeline.py --dry-run
  
  # Run full pipeline
  python run_pipeline.py --run
  
  # Force re-run all stages
  python run_pipeline.py --run --force
  
  # Run single stage
  python run_pipeline.py --stage data_validation
  
  # Show metrics
  python run_pipeline.py --metrics
        """
    )
    
    parser.add_argument('--dag', action='store_true', help='Show pipeline DAG')
    parser.add_argument('--status', action='store_true', help='Show pipeline status')
    parser.add_argument('--dry-run', action='store_true', help='Run pipeline dry-run')
    parser.add_argument('--run', action='store_true', help='Run full pipeline')
    parser.add_argument('--stage', type=str, help='Run specific stage')
    parser.add_argument('--force', action='store_true', help='Force re-run stages')
    parser.add_argument('--metrics', action='store_true', help='Show pipeline metrics')
    parser.add_argument('--params', action='store_true', help='Show pipeline parameters')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    success = True
    
    try:
        if args.dag:
            success = show_pipeline_dag()
        
        if args.status:
            success = show_pipeline_status() and success
        
        if args.params:
            success = show_params() and success
        
        if args.dry_run:
            success = run_pipeline_dry_run() and success
        
        if args.run:
            success = run_full_pipeline(force=args.force) and success
        
        if args.stage:
            success = run_single_stage(args.stage, force=args.force) and success
        
        if args.metrics:
            success = show_pipeline_metrics() and success
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Pipeline runner failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
