import sys
import os
import pytest
import pandas as pd
import numpy as np

# --------------------------------------------------
# Ensure project root is in PYTHONPATH
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# --------------------------------------------------
# Pytest global configuration
# --------------------------------------------------
def pytest_configure():
    """
    Force MLflow to use a local file-based store during tests.
    This avoids Alembic + SQLite migration issues in CI.
    """
    os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlruns"
    os.environ["MLFLOW_ARTIFACT_URI"] = "file:///tmp/mlruns"


# --------------------------------------------------
# Fixtures
# --------------------------------------------------
@pytest.fixture
def sample_raw_data():
    """Creates a sample raw dataframe mimicking the UCI heart disease dataset."""
    data = {
        "age": [63, 37, 41, 56],
        "sex": [1, 1, 0, 1],
        "cp": [3, 2, 1, 1],
        "trestbps": [145, 130, 130, 120],
        "chol": [233, 250, 204, 236],
        "fbs": [1, 0, 0, 0],
        "restecg": [0, 1, 0, 1],
        "thalach": [150, 187, 172, 178],
        "exang": [0, 0, 0, 0],
        "oldpeak": [2.3, 3.5, 1.4, 0.8],
        "slope": [0, 0, 2, 2],
        "ca": ["0", "0", "0", "?"],   # Intentionally include '?'
        "thal": ["1", "2", "2", "2"],
        "target": [1, 1, 1, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_prepared_data(sample_raw_data):
    """Creates a sample dataframe that looks like the output of data preparation."""
    df = sample_raw_data.copy()

    # Simulate cleaning
    df.replace("?", np.nan, inplace=True)
    df["ca"] = pd.to_numeric(df["ca"])

    return df
