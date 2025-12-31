import unittest
import pandas as pd
import sys
import os

# Add scripts directory to path to import generate_data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from generate_data import generate_churn_data

class TestGenerateData(unittest.TestCase):
    def test_generate_churn_data_shape(self):
        n_samples = 100
        df = generate_churn_data(n_samples=n_samples, random_seed=42)
        self.assertEqual(len(df), n_samples)
        self.assertEqual(len(df.columns), 7)
    
    def test_generate_churn_data_columns(self):
        df = generate_churn_data(n_samples=10, random_seed=42)
        expected_columns = [
            'customer_id', 'tenure_months', 'num_logins_last_30d', 
            'num_tickets_last_90d', 'plan_type', 'country', 'is_churned'
        ]
        self.assertListEqual(list(df.columns), expected_columns)
    
    def test_is_churned_values(self):
        df = generate_churn_data(n_samples=100, random_seed=42)
        # Check if is_churned only contains 0 and 1
        unique_values = df['is_churned'].unique()
        for val in unique_values:
            self.assertIn(val, [0, 1])

if __name__ == '__main__':
    unittest.main()
