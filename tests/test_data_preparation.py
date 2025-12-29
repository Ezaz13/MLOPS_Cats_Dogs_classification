import unittest
import pandas as pd
import numpy as np
from src.data_preparation.preparation import clean_data

class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        """Creates a sample raw dataframe mimicking the UCI heart disease dataset."""
        data = {
            'age': [63, 37, 41, 56],
            'sex': [1, 1, 0, 1],
            'cp': [3, 2, 1, 1],
            'trestbps': [145, 130, 130, 120],
            'chol': [233, 250, 204, 236],
            'fbs': [1, 0, 0, 0],
            'restecg': [0, 1, 0, 1],
            'thalach': [150, 187, 172, 178],
            'exang': [0, 0, 0, 0],
            'oldpeak': [2.3, 3.5, 1.4, 0.8],
            'slope': [0, 0, 2, 2],
            'ca': ['0', '0', '0', '?'],  # Intentionally include '?'
            'thal': ['1', '2', '2', '2'],
            'target': [1, 1, 1, 0]
        }
        self.sample_raw_data = pd.DataFrame(data)

    def test_clean_data_replaces_question_marks(self):
        """Test that '?' values are replaced with NaN."""
        # Ensure our setup has a '?'
        self.assertIn('?', self.sample_raw_data['ca'].values)
        
        cleaned_df = clean_data(self.sample_raw_data)
        
        # Check if '?' is gone
        self.assertNotIn('?', cleaned_df['ca'].values)
        # Check if it was replaced by NaN or coerced properly
        self.assertTrue(cleaned_df['ca'].isnull().sum() > 0 or np.issubdtype(cleaned_df['ca'].dtype, np.number))

    def test_clean_data_numeric_conversion(self):
        """Test that numeric columns stored as objects are converted to numeric types."""
        cleaned_df = clean_data(self.sample_raw_data)
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['ca']))

    def test_clean_data_target_normalization(self):
        """Test that target variable is normalized to 0 and 1."""
        self.sample_raw_data.loc[0, 'target'] = 3
        cleaned_df = clean_data(self.sample_raw_data)
        unique_targets = cleaned_df['target'].unique()
        self.assertTrue(set(unique_targets).issubset({0, 1}))