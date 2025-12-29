import unittest
import pandas as pd
import numpy as np
from src.data_transformation_and_storage.transformation import perform_feature_engineering

class TestTransformation(unittest.TestCase):
    def setUp(self):
        """Creates a sample dataframe that looks like the output of data preparation."""
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
            'ca': [0.0, 0.0, 0.0, np.nan], # Simulating cleaned data
            'thal': ['1', '2', '2', '2'],
            'target': [1, 1, 1, 0]
        }
        self.sample_prepared_data = pd.DataFrame(data)

    def test_feature_engineering_creates_rpp(self):
        """Test creation of Rate Pressure Product."""
        df_transformed = perform_feature_engineering(self.sample_prepared_data)
        
        self.assertIn('rate_pressure_product', df_transformed.columns)
        expected = self.sample_prepared_data['thalach'] * self.sample_prepared_data['trestbps']
        pd.testing.assert_series_equal(df_transformed['rate_pressure_product'], expected, check_names=False)

    def test_feature_engineering_creates_age_groups(self):
        """Test creation of age group one-hot encoded columns."""
        df_transformed = perform_feature_engineering(self.sample_prepared_data)
        age_cols = [col for col in df_transformed.columns if 'age_q' in col]
        self.assertTrue(len(age_cols) > 0)

    def test_feature_engineering_high_risk_flag(self):
        """Test creation of is_high_risk flag."""
        df_transformed = perform_feature_engineering(self.sample_prepared_data)
        self.assertIn('is_high_risk', df_transformed.columns)
        self.assertTrue(df_transformed['is_high_risk'].isin([0, 1]).all())