import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock  
from app.scripts.train import preprocess_data, train_model, evaluate_model, load_data

class TestTrain(unittest.TestCase):
    def setUp(self):
        # Create a small dummy dataset
        self.df = pd.DataFrame({
            'customer_id': [f'C_{i}' for i in range(20)],
            'tenure_months': np.random.randint(1, 100, 20),
            'num_logins_last_30d': np.random.randint(0, 30, 20),
            'num_tickets_last_90d': np.random.randint(0, 10, 20),
            'plan_type': np.random.choice(['basic', 'premium'], 20),
            'country': np.random.choice(['US', 'UK'], 20),
            'is_churned': np.random.randint(0, 2, 20)
        })

    @patch('pandas.read_parquet')
    def test_load_data_parquet(self, mock_read_parquet):
        mock_read_parquet.return_value = self.df
        df_loaded = load_data('dummy.parquet')
        self.assertEqual(len(df_loaded), 20)
        mock_read_parquet.assert_called_once_with('dummy.parquet')

    def test_preprocess_data(self):
        X, y, encoders = preprocess_data(self.df)
        self.assertEqual(X.shape, (20, 5))
        self.assertEqual(len(y), 20)
        self.assertIn('plan_type', encoders)
        self.assertIn('country', encoders)
        # Check if categorical are numeric now
        self.assertTrue(np.issubdtype(X['plan_type'].dtype, np.number))

    def test_train_model(self):
        X, y, _ = preprocess_data(self.df)
        model, params = train_model(X, y, n_estimators=10, max_depth=5)
        self.assertIsNotNone(model)
        self.assertEqual(params['n_estimators'], 10)

    def test_evaluate_model(self):
        X, y, _ = preprocess_data(self.df)
        model, _ = train_model(X, y, n_estimators=10, max_depth=5)
        metrics, y_pred, y_proba = evaluate_model(model, X, y)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        self.assertEqual(len(y_pred), 20)
        self.assertEqual(len(y_proba), 20)

if __name__ == '__main__':
    unittest.main()
