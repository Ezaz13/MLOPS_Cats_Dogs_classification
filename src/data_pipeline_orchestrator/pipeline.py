"""
Data Pipeline Orchestrator using Prefect.

This script defines and executes the entire data pipeline as a Prefect flow.
Each step in the pipeline is a Prefect task, and the flow function defines the
Directed Acyclic Graph (DAG) of task dependencies.

To run this pipeline, ensure you have Prefect installed:
  pip install prefect

Then, execute this script from the project root:
  python src/data_pipeline_orchestrator/pipeline.py
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Any
from prefect import task, flow, get_run_logger

# Define the project root to ensure all paths are consistent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

@task(retries=1, retry_delay_seconds=5, name="Run Pipeline Step")
def run_script(script_path: Path, task_name: str, upstream_result: Any = None) -> bool:
    """
    A Prefect task to run a Python script as a subprocess.
    This task represents a single node in our DAG.
    """
    logger = get_run_logger()
    logger.info(f"--- Starting Task: {task_name} ---")
    logger.info(f"Executing script: {script_path}")

    if not script_path.exists():
        logger.error(f"!!! Task '{task_name}' FAILED: Script not found at {script_path} !!!")
        raise FileNotFoundError(f"Script not found at {script_path}")

    # Force the subprocess to use UTF-8 for its own console output.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    try:
        # Run the script without capturing output. Let it print directly to the console.
        # `check=True` will still raise an error if the script fails (returns a non-zero exit code).
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            check=True,
            env=env
        )
        logger.info(f"--- Task '{task_name}' completed successfully. ---")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"!!! Task '{task_name}' FAILED with exit code {e.returncode} !!!")
        # No need to log stderr as it will have already been printed to the console.
        raise  # Re-raise the exception to make the Prefect task fail

@flow(name="MLOPS End-to-End Pipeline")
def mlops_pipeline():
    """
    The main Prefect flow that orchestrates the entire data pipeline.
    This function defines the DAG by calling tasks sequentially.
    """
    logger = get_run_logger()
    logger.info("====== Starting Data Pipeline Run with Prefect ======")

    # Define the paths to the scripts for each task
    ingestion_script = PROJECT_ROOT / "src" / "data_ingestion" / "ingestion.py"
    validation_script = PROJECT_ROOT / "src" / "data_validation" / "validation.py"
    preparation_script = PROJECT_ROOT / "src" / "data_preparation" / "preparation.py"
    transformation_script = PROJECT_ROOT / "src" / "data_transformation" / "transformation.py"
    model_building_script = PROJECT_ROOT / "src" / "model_building" / "train_model.py"

    # Execute the DAG sequentially. Prefect waits for each task to complete.
    ingestion_result = run_script(script_path=ingestion_script, task_name="Data Ingestion")
    validation_result = run_script(script_path=validation_script, task_name="Data Validation", upstream_result=ingestion_result)
    preparation_result = run_script(script_path=preparation_script, task_name="Data Preparation", upstream_result=validation_result)
    transformation_result = run_script(script_path=transformation_script, task_name="Data Transformation", upstream_result=preparation_result)
    model_building_result = run_script(script_path=model_building_script, task_name="Model Building", upstream_result=transformation_result)

    logger.info("====== Prefect Pipeline Flow Finished Successfully ======")

if __name__ == "__main__":
    mlops_pipeline()
    print("\nPipeline execution complete. Check the logs for task status.")
    print("For a richer UI and history, serve your flows with 'prefect server start' and deploy your flow.")
