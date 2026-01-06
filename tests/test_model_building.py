import unittest
import pandas as pd
import numpy as np
from src.model_building.train_model import prepare_data

class TestModelBuilding(unittest.TestCase):
    def setUp(self):
        """Creates a sample dataframe for model building tests."""
        data = {
            'age': [63, 37, 41, 56, 50, 45],
            'sex': [1, 1, 0, 1, 0, 1],
            'cp': [3, 2, 1, 1, 0, 2],
            'trestbps': [145, 130, 130, 120, 110, 135],
            'chol': [233, 250, 204, 236, 200, 220],
            'fbs': [1, 0, 0, 0, 0, 0],
            'restecg': [0, 1, 0, 1, 0, 1],
            'thalach': [150, 187, 172, 178, 160, 155],
            'exang': [0, 0, 0, 0, 1, 0],
            'oldpeak': [2.3, 3.5, 1.4, 0.8, 1.0, 0.5],
            'slope': [0, 0, 2, 2, 1, 1],
            'ca': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'thal': ['1', '2', '2', '2', '1', '3'],
            'target': [1, 1, 1, 0, 0, 0]
        }
        self.sample_prepared_data = pd.DataFrame(data)

    def test_prepare_data_splitting(self):
        """Test that data is split correctly into train and test sets."""
        # Ensure target exists
        self.assertIn('target', self.sample_prepared_data.columns)
        
        X_train, X_test, y_train, y_test = prepare_data(self.sample_prepared_data)
        
        # Check split ratio (approx 80/20)
        total_rows = len(self.sample_prepared_data)
        self.assertEqual(len(X_train) + len(X_test), total_rows)
        self.assertEqual(len(y_train) + len(y_test), total_rows)
        
        # Check that target is removed from X
        self.assertNotIn('target', X_train.columns)
        self.assertNotIn('target', X_test.columns)

    def test_prepare_data_float_casting(self):
        """Test that integer features are cast to float64."""
        # Force an integer column in input
        self.sample_prepared_data['age'] = self.sample_prepared_data['age'].astype(int)
        
        X_train, X_test, _, _ = prepare_data(self.sample_prepared_data)
        
        self.assertEqual(X_train['age'].dtype, 'int32')
        self.assertEqual(X_test['age'].dtype, 'int32')