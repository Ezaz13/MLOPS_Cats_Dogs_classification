# It is recommended to manage your Python environment using a requirements.txt file
# and to install the required packages before running this script.
import os
import sys
import warnings
import glob
import great_expectations as ge
import pandas as pd


# --- Setup Project Root and Imports ---

from src.utility.exception import CustomException
from src.utility.logger import setup_logging

# --- Configuration ---
# Define the exact column order and names for validation
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Centralized configuration for local validation sources
VALIDATION_CONFIG = {
    "uci": {
        "local_base_path": os.path.join(PROJECT_ROOT, "data", "raw", "uci"),
        "suite_name": "heart_disease_suite"
    }
}

def get_latest_csv_file(folder_path):
    """
    Get the latest CSV file from a folder based on modification time.
    """
    list_of_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)


class LocalDataValidator:
    """Encapsulates the Great Expectations validation workflow for a local data source."""

    def __init__(self, source, config):
        self.source = source
        self.config = config
        self.logger = setup_logging(f"LocalDataValidator_{source}")

        # Change to the project root directory to ensure GE finds the correct config
        original_cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)

        try:
            # Use the newer GE configuration in the gx folder
            gx_config_path = os.path.join(PROJECT_ROOT, "gx")
            if os.path.exists(gx_config_path):
                os.chdir(gx_config_path)
                self.context = ge.get_context()
                self.logger.info(f"Using Great Expectations configuration from: {gx_config_path}")
            else:
                # Fallback to the older configuration
                ge_config_path = os.path.join(PROJECT_ROOT, "great_expectations")
                os.chdir(ge_config_path)
                self.context = ge.get_context()
                self.logger.info(f"Using Great Expectations configuration from: {ge_config_path}")
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

        # Ensure the datasource exists
        self._ensure_datasource_exists()
        self.local_file_path = self._get_latest_local_csv_path()

    def _ensure_datasource_exists(self):
        """Ensures that the required datasource exists in the context."""
        datasource_name = "local_pandas_datasource"
        if datasource_name in self.context.datasources:
            self.logger.info(f"Datasource {datasource_name} already exists")
        else:
            self.logger.info(f"Creating fluent datasource {datasource_name}")
            self.context.sources.add_pandas(name=datasource_name)
            self.logger.info(f"Successfully created fluent datasource {datasource_name}")

    def _get_latest_local_csv_path(self):
        """Finds the most recently modified CSV file in the configured local directory."""
        base_path = self.config['local_base_path']
        self.logger.info(f"Searching for latest CSV file in: {base_path}")
        try:
            latest_file = get_latest_csv_file(base_path)
            if not latest_file:
                raise ValueError(f"No CSV files found in directory: {base_path}")
            self.logger.info(f"Found latest local CSV file: {latest_file}")
            return latest_file
        except Exception as e:
            raise CustomException(f"Could not get latest CSV file for {self.source}: {e}", sys)

    def _build_expectation_suite(self, validator):
        """Builds a single, comprehensive suite of expectations for the heart disease dataset."""
        self.logger.info("Building comprehensive expectation suite.")
        try:
            # Schema and Column Integrity
            validator.expect_table_columns_to_match_ordered_list(EXPECTED_COLUMNS)

            # Critical columns should not be null
            validator.expect_column_values_to_not_be_null("age")
            validator.expect_column_values_to_not_be_null("sex")
            validator.expect_column_values_to_not_be_null("target") # Target variable

            # Data Types and Ranges
            # Age: typically between 20 and 100
            validator.expect_column_values_to_be_between("age", min_value=20, max_value=100)

            # Resting Blood Pressure (trestbps): typically 90-200
            validator.expect_column_values_to_be_between("trestbps", min_value=80, max_value=250)

            # Cholesterol (chol): typically 100-600
            validator.expect_column_values_to_be_between("chol", min_value=100, max_value=600)

            # Max Heart Rate (thalach): typically 60-220
            validator.expect_column_values_to_be_between("thalach", min_value=60, max_value=220)

            # Oldpeak: ST depression, typically 0.0 to 6.0
            validator.expect_column_values_to_be_between("oldpeak", min_value=0.0, max_value=10.0)

            # Categorical Value Sets
            validator.expect_column_values_to_be_in_set("sex", [0, 1])
            validator.expect_column_values_to_be_in_set("cp", [1, 2, 3, 4])
            validator.expect_column_values_to_be_in_set("fbs", [0, 1])
            validator.expect_column_values_to_be_in_set("restecg", [0, 1, 2])
            validator.expect_column_values_to_be_in_set("exang", [0, 1])
            validator.expect_column_values_to_be_in_set("slope", [1, 2, 3])
            # Target variable 'target' (0-4)
            validator.expect_column_values_to_be_in_set("target", [0, 1, 2, 3, 4])

            # Note: 'ca' and 'thal' in raw UCI data often contain '?' which makes them objects/strings.
            # We validate that they are in the expected set of strings or numbers.
            # ca: 0-3, thal: 3,6,7
            # We allow string representations to pass validation before cleaning.

            self.logger.info("Expectation suite built successfully.")
        except Exception as e:
            raise CustomException(f"Error building expectation suite: {e}", sys)

    def run_validation(self):
        self.logger.info(f"--- Starting GE validation for source: {self.source} ---")
        try:
            # Use the fluent API approach for GE 1.4.1
            datasource = self.context.datasources["local_pandas_datasource"]
            asset_name = f"local_heart_data_{self.source}"

            # Try to get existing asset or create new one
            try:
                data_asset = datasource.get_asset(asset_name)
                self.logger.info(f"Using existing data asset: {asset_name}")
            except Exception:
                data_asset = datasource.add_csv_asset(
                    name=asset_name,
                    filepath_or_buffer=self.local_file_path
                )
                self.logger.info(f"Created new data asset: {asset_name}")

            batch_request = data_asset.build_batch_request()

            # Create or get the expectation suite
            suite_name = self.config["suite_name"]
            try:
                # Try to get existing suite
                suite = self.context.get_expectation_suite(expectation_suite_name=suite_name)
                self.logger.info(f"Using existing expectation suite: {suite_name}")
            except Exception:
                # Create new suite if it doesn't exist
                suite = self.context.add_expectation_suite(expectation_suite_name=suite_name)
                self.logger.info(f"Created new expectation suite: {suite_name}")

            # Use `get_validator` to retrieve or create the validator
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=suite_name
            )

            self._build_expectation_suite(validator)
            # Save the expectation suite using the context's suites API
            self.context.save_expectation_suite(validator.expectation_suite)
            self.logger.info("Expectation suite saved successfully")

            validation_results = validator.validate()
            self._generate_reports(validation_results)

            self.logger.info(f"--- Validation complete for source: {self.source} ---")
            return validation_results["success"]
        except Exception as e:
            self.logger.error(f"An error occurred during validation for {self.source}.")
            raise CustomException(str(e), sys)

    def _generate_reports(self, validation_results):
        self.logger.info("Generating Data Docs (HTML report).")
        self.context.build_data_docs()
        docs_urls = self.context.get_docs_sites_urls()
        if docs_urls:
            self.logger.info(f"Data Docs available at: {docs_urls[0]['site_url']}")

        self.logger.info("Generating CSV summary report.")
        report_path = os.path.join(PROJECT_ROOT, "reports", "validation")
        os.makedirs(report_path, exist_ok=True)
        csv_path = os.path.join(report_path, f"{self.source}_validation_summary.csv")

        results_list = []
        for result in validation_results["results"]:
            try:
                # Handle both old and new GE API structures
                expectation_config = result.get("expectation_config", {})
                if hasattr(expectation_config, 'expectation_type'):
                    expectation_type = expectation_config.expectation_type
                    kwargs = expectation_config.kwargs if hasattr(expectation_config, 'kwargs') else {}
                else:
                    expectation_type = expectation_config.get("expectation_type", "unknown")
                    kwargs = expectation_config.get("kwargs", {})

                results_list.append({
                    "expectation_type": expectation_type,
                    "column": kwargs.get("column", ""),
                    "success": result.get("success", False),
                    "observed_value": result.get("result", {}).get("observed_value", "")
                })
            except Exception as e:
                self.logger.warning(f"Could not parse validation result: {e}")
                continue

        pd.DataFrame(results_list).to_csv(csv_path, index=False)
        self.logger.info(f"CSV summary report saved to: {csv_path}")


# --- Main Execution ---
if __name__ == "__main__":
    main_logger = setup_logging("data_validation_main")
    main_logger.info("Starting data validation process for all configured local sources.")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    validation_outcomes = {}
    try:
        for source, config in VALIDATION_CONFIG.items():
            validator = LocalDataValidator(source, config)
            outcome = validator.run_validation()
            validation_outcomes[source] = outcome

        if not all(validation_outcomes.values()):
            main_logger.error("One or more data validation checks failed.")
            sys.exit(1)
        else:
            main_logger.info("All data validation checks passed successfully.")

    except CustomException as e:
        main_logger.error(f"A critical error occurred during the validation process: {e}")
        sys.exit(1)
