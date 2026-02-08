import pytest
import pandas as pd
from src.data_preparation import preparation

class TestDataPreparation:
    def test_load_dataset(self, mock_dataset, monkeypatch):
        """Test loading images from the raw data directory."""
        # Mock the RAW_DATA_PATH in the module
        monkeypatch.setattr(preparation, "RAW_DATA_PATH", mock_dataset / "data" / "raw" / "PetImages")
        
        df = preparation.load_dataset()
        
        # We created 3 valid Cat images + 1 valid Dog image + 1 corrupt Dog in fixture
        # Wait, check conftest logic:
        # cat1, cat2, dog1 = valid (RGB)
        # dog_corrupt = corrupt
        # load_dataset relies on VALID_EXTENSIONS.
        # It blindly lists files. So it should load ALL 4 files.
        
        assert len(df) == 4
        assert set(df["label"].unique()) == {"cat", "dog"}
        assert "filepath" in df.columns

    def test_clean_data(self, mock_dataset, monkeypatch):
        """Test that clean_data removes corrupt images."""
        monkeypatch.setattr(preparation, "RAW_DATA_PATH", mock_dataset / "data" / "raw" / "PetImages")
        
        # Load first (get all 4)
        df_raw = preparation.load_dataset()
        
        # Run clean_data
        df_clean = preparation.clean_data(df_raw)
        
        # Should remove the 1 corrupt image
        assert len(df_clean) == 3
        
        # Check that we have dimensions
        assert "width" in df_clean.columns
        assert "height" in df_clean.columns
        
        # Verify the corrupt one is gone
        for filepath in df_clean["filepath"]:
            assert "corrupt" not in filepath

    def test_validate_class_distribution(self, mock_dataset, monkeypatch):
        """Test class distribution validation."""
        monkeypatch.setattr(preparation, "RAW_DATA_PATH", mock_dataset / "data" / "raw" / "PetImages")
        
        # Create a balanced df manually or use the cleaned one
        # Our fixture has 2 cats, 1 dog (valid). This is imbalanced but valid (ratio 2:1 <= 2).
        
        df_clean = preparation.clean_data(preparation.load_dataset())
        
        # Should NOT raise exception
        preparation.validate_class_distribution(df_clean)
        
        # Now create a severely imbalanced df
        # 10 cats, 1 dog
        data = {
            "label": ["cat"] * 10 + ["dog"],
            "filepath": ["dummy"] * 11
        }
        df_imbalanced = pd.DataFrame(data)
        
        # The function logs a warning but doesn't raise exception unless count is 0.
        # Let's verify it raises if a class is MISSING.
        
        df_missing = pd.DataFrame({"label": ["cat"] * 10, "filepath": ["dummy"] * 10})
        
        with pytest.raises(Exception) as excinfo:
            preparation.validate_class_distribution(df_missing)
        
        assert "No samples found for class 'dog'" in str(excinfo.value)