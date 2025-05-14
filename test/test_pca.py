import numpy as np
import sys, os
proj_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
import unittest
from PyOD_ADGE.models.principal_component_analysis import PCA  

class TestPCA(unittest.TestCase):
    def setUp(self):
        # Dataset sintético simple para cada test
        self.X = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [10.0, 10.0, 10.0],  # posible outlier
        ])

    def test_fit_sets_attributes(self):
        model = PCA(n_components=2, n_selected_components=1, contamination=0.25)
        model.fit(self.X)

        self.assertTrue(hasattr(model, "components_"))
        self.assertTrue(hasattr(model, "explained_variance_"))
        self.assertTrue(hasattr(model, "w_components_"))
        self.assertTrue(hasattr(model, "selected_components_"))
        self.assertTrue(hasattr(model, "decision_scores_"))

    def test_decision_function_output_shape(self):
        model = PCA(n_components=2, n_selected_components=1)
        model.fit(self.X)
        scores = model.decision_function(self.X)
        self.assertEqual(scores.shape, (self.X.shape[0],))
        self.assertTrue(np.all(scores >= 0))

    def test_without_standardization(self):
        model = PCA(n_components=2, n_selected_components=1, standardization=False)
        model.fit(self.X)
        scores = model.decision_function(self.X)
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(scores.shape[0], self.X.shape[0])

    def test_weighted_vs_unweighted(self):
        model_w = PCA(n_components=2, n_selected_components=1, weighted=True)
        model_u = PCA(n_components=2, n_selected_components=1, weighted=False)
        model_w.fit(self.X)
        model_u.fit(self.X)
        scores_w = model_w.decision_function(self.X)
        scores_u = model_u.decision_function(self.X)
        # Deben diferir cuando weighted cambia
        self.assertFalse(np.allclose(scores_w, scores_u))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            PCA(n_components=-1).fit(self.X)
        with self.assertRaises(ValueError):
            PCA(contamination=0.6).fit(self.X)
        with self.assertRaises(ValueError):
            PCA(n_components=None, n_selected_components=None).fit(self.X)

if __name__ == '__main__':
    print("Running PCA tests...")
    unittest.main()