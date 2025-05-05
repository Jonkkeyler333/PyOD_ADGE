import sys, os
proj_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
import unittest
import numpy as np
from PyOD_ADGE.models.local_outlier_factor import LocalOutlierFactor

class TestLocalOutlierFactor(unittest.TestCase):

    def test_init_validations(self):
        # k_neighbors < 1
        with self.assertRaises(ValueError):
            LocalOutlierFactor(k_neighbors=0)
        # contamination fuera de rango
        with self.assertRaises(ValueError):
            LocalOutlierFactor(contamination=-0.1)
        with self.assertRaises(ValueError):
            LocalOutlierFactor(contamination=0.6)

    def test_unknown_metric_raises_on_fit(self):
        # La validación de la métrica se hace en fit(), no en __init__
        lof = LocalOutlierFactor(metric='unknown')
        X = np.random.rand(10, 2)
        with self.assertRaises(ValueError):
            lof.fit(X)

    def test_fit_too_few_samples(self):
        lof = LocalOutlierFactor(k_neighbors=5)
        X = np.random.rand(4, 2)  # menos de k_neighbors
        with self.assertRaises(ValueError):
            lof.fit(X)

    def test_k_distance_and_neighbors(self):
        X = np.array([
            [0, 0],
            [1, 0],
            [0.5, np.sqrt(3)/2]
        ])
        lof = LocalOutlierFactor(k_neighbors=1)
        lof.fit(X)
        # Cada punto debe tener k_distance ≈ 1.0
        for k_distance, neighbors in lof._k_distances_cache.values():
            self.assertAlmostEqual(k_distance, 1.0, places=6)
            # Y al menos k_neighbors vecinos (aquí serán 2, pues ambos están a distancia 1)
            self.assertGreaterEqual(len(neighbors), lof.k_neighbors)

    def test_local_reachability_density_and_lof(self):
        # Dataset: dos puntos muy cercanos y uno aislado
        close = np.array([[0, 0], [0, 0.01]])
        outlier = np.array([[1, 1]])
        X = np.vstack([close, outlier])
        lof = LocalOutlierFactor(k_neighbors=1, contamination=1/3)
        lof.fit(X)
        scores = lof.outlier_factor_
        # El outlier (índice 2) debe tener LOF mayor que los cercanos
        self.assertTrue(scores[2] > scores[0])
        self.assertTrue(scores[2] > scores[1])

    def test_decision_function_labels(self):
        close = np.array([[0, 0], [0, 0.01]])
        outlier = np.array([[1, 1]])
        X = np.vstack([close, outlier])
        lof = LocalOutlierFactor(k_neighbors=1, contamination=1/3)
        lof.fit(X)
        labels = lof.decision_function()
        # Con 3 muestras y contamination=1/3, debe marcar 1 como -1 (outlier)
        self.assertEqual(list(labels).count(-1), 1)
        self.assertEqual(list(labels).count(1), 2)

if __name__ == '__main__':
    print("Running tests for LocalOutlierFactor...")
    unittest.main()
