import sys, os
proj_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
import unittest
import numpy as np
from sklearn.datasets import make_blobs
from PyOD_ADGE.models.feature_bagging import FeatureBagging 


class TestFeatureBagging(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(n_samples=100, n_features=5, random_state=42)
        self.default_params = {
            'n_estimators': 5,
            'random_state': 42,
            'contamination': 0.1,
            'n_jobs': 1
        }

    def test_init_validation(self):
        with self.assertRaises(ValueError):
            FeatureBagging(base_estimator='invalid')
            
        with self.assertRaises(ValueError):
            FeatureBagging(combine='invalid')

        with self.assertRaises(ValueError):
            FeatureBagging(n_estimators=0)

        with self.assertRaises(ValueError):
            FeatureBagging(contamination=0.6)

    def test_fit(self):
        model = FeatureBagging(**self.default_params)
        model.fit(self.X)
        
        self.assertTrue(hasattr(model, '_is_fitted'))
        self.assertEqual(len(model.subsets_), self.default_params['n_estimators'])
        self.assertEqual(len(model.estimators_scores_), self.default_params['n_estimators'])

    def test_decision_function_shape(self):
        model = FeatureBagging(**self.default_params).fit(self.X)
        scores = model.decision_function()
        self.assertEqual(scores.shape, (self.X.shape[0],))

    def test_outlier_factor_property(self):
        model = FeatureBagging(combine='cumulative', **self.default_params).fit(self.X)
        scores = model.outlier_factor_
        self.assertEqual(scores.shape, (self.X.shape[0],))

    def test_bagging_features(self):
        model = FeatureBagging(**self.default_params)
        d = 10
        rng = np.random.default_rng(42)
        features = model._bagging_features_idx(d, rng)
        self.assertTrue(len(features) >= d//2)
        self.assertTrue(len(features) <= d)
        self.assertEqual(len(np.unique(features)), len(features))

    def test_combine_methods(self):
        model = FeatureBagging(combine='breadth', **self.default_params).fit(self.X)
        self.assertTrue(hasattr(model, 'threshold_'))
        model = FeatureBagging(combine='cumulative', **self.default_params).fit(self.X)
        self.assertEqual(model.outlier_factor_.shape, (self.X.shape[0],))
        self.assertLessEqual(model.threshold_, model.outlier_factor_.max())

    def test_parallel_processing(self):
        model = FeatureBagging(n_jobs=2, **self.default_params).fit(self.X)
        self.assertTrue(model._is_fitted)

if __name__ == '__main__':
    unittest.main()