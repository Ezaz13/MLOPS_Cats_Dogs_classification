import sys
import os
import shutil
import pytest
import numpy as np
from PIL import Image
from pathlib import Path

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
def temp_project_structure(tmp_path):
    """
    Creates a temporary project structure with raw data folders.
    Returns the root path of the temp project.
    """
    data_root = tmp_path / "data" / "raw" / "PetImages"
    (data_root / "Cat").mkdir(parents=True)
    (data_root / "Dog").mkdir(parents=True)
    
    # Create prepared data dir
    (tmp_path / "data" / "prepared").mkdir(parents=True)
    return tmp_path

@pytest.fixture
def sample_image():
    """Creates a simple RGB dummy image."""
    return Image.new('RGB', (100, 100), color='red')

@pytest.fixture
def corrupt_image():
    """Creates a dummy text file pretending to be an image (corrupt)."""
    return b"This is not an image."

@pytest.fixture
def mock_dataset(temp_project_structure, sample_image, corrupt_image):
    """
    Populates the temp project structure with:
    - 2 valid Cat images
    - 1 valid Dog image
    - 1 corrupt Dog image
    """
    cat_dir = temp_project_structure / "data" / "raw" / "PetImages" / "Cat"
    dog_dir = temp_project_structure / "data" / "raw" / "PetImages" / "Dog"

    # Save valid images
    sample_image.save(cat_dir / "cat1.jpg")
    sample_image.save(cat_dir / "cat2.jpg")
    sample_image.save(dog_dir / "dog1.jpg")

    # Save corrupt image
    with open(dog_dir / "dog_corrupt.jpg", "wb") as f:
        f.write(corrupt_image)

    return temp_project_structure
