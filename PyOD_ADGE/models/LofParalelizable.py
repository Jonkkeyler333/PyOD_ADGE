from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted
import scipy.spatial.distance
from joblib import Parallel, delayed
import numpy as np

class LocalOutlierFactorParalelizable(BaseEstimator, OutlierMixin):
    def __init__(self, k_neighbors: int = 20, metric: str = 'euclidean', contamination: float = 0.1, n_jobs=-1, **metric_params):
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be greater than 0")
        if not (0 <= contamination <= 0.5):
            raise ValueError("contamination must be between 0 and 0.5")
        if not isinstance(n_jobs, int):
            raise ValueError("n_jobs must be an integer.")

        self.k_neighbors = k_neighbors
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.contamination = contamination
        self._k_distances_cache = {}

    def fit(self, X, y=None):
        self._X = X
        if X.shape[0] < self.k_neighbors:
            raise ValueError("The number of samples must be greater than k_neighbors")
        self.metric_c = self._get_distance_metric()
        self._k_distances_cache = self._get_all_k_distances(self._X)
        self.decision_scores_ = self._local_outlier_factor(self._X)
        self._is_fitted = True
        return self


    def decision_function(self):
        scores = self.outlier_factor_
        self.threshold_ = np.quantile(scores, 1 - self.contamination)
        return np.where(scores > self.threshold_, -1, 1)
        
    def predict(self):
        check_is_fitted(self, '_is_fitted')
        return self.outlier_factor_


    @property
    def outlier_factor_(self) -> np.ndarray:
        check_is_fitted(self, '_is_fitted')
        sorted_scores = sorted(self.decision_scores_, key=lambda x: x[0])
        return np.array([score[1] for score in sorted_scores])

    def _get_distance_metric(self):
        if self.metric == 'minkowski':
            return lambda u, v: scipy.spatial.distance.minkowski(u, v, **self.metric_params)
        metrics = {
            'euclidean': scipy.spatial.distance.euclidean,
            'manhattan': scipy.spatial.distance.cityblock,
            'cosine': scipy.spatial.distance.cosine,
            'chebyshev': scipy.spatial.distance.chebyshev,
        }
        try:
            return metrics[self.metric]
        except KeyError:
            raise ValueError(f"Unknown metric: {self.metric}.")

    def _k_distances(self, D, p, idx_p):
        distances = []
        for idx, o in enumerate(D):
            if idx == idx_p:
                continue
            distances.append((self.metric_c(o, p), idx))
        distances.sort(key=lambda x: x[0])
        k_distance = distances[self.k_neighbors - 1][0]
        k_neighbors = [idx for dist, idx in distances if dist <= k_distance]
        return k_distance, k_neighbors

    def _get_all_k_distances(self, D) -> dict:
        def _process_point(idx_p):
            k_distance, k_neighbors = self._k_distances(D, D[idx_p], idx_p)
            return idx_p, (k_distance, k_neighbors)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_process_point)(idx_p) for idx_p in range(len(D))
        )
        return dict(results)

    def _reach_dist(self, p, o, k_distance):
        return max(k_distance, self.metric_c(p, o))

    def _local_reachability_density(self, D) -> list:
        k_distances = self._k_distances_cache

        def _compute_lrd(idx_p):
            p = D[idx_p]
            _, k_neighbors_p = k_distances[idx_p]
            reach_distances = [
                self._reach_dist(p, D[idx_o], k_distances[idx_o][0])
                for idx_o in k_neighbors_p
            ]
            srd = np.sum(reach_distances)
            if srd == 0:
                srd = np.inf
            lrd_value = 1 / (srd / len(k_neighbors_p))
            return (idx_p, lrd_value)

        lrd = Parallel(n_jobs=self.n_jobs)(
            delayed(_compute_lrd)(idx_p) for idx_p in range(len(D))
        )
        return lrd

    def _local_outlier_factor(self, D):
        lrd = self._local_reachability_density(D)
        lrd_dict = dict(lrd)

        def _compute_lof(idx_p):
            _, k_neighbors_p = self._k_distances_cache[idx_p]
            lrd_p = lrd_dict[idx_p]
            sum_lrd = 0
            for idx_o in k_neighbors_p:
                lrd_o = lrd_dict[idx_o]
                if np.isinf(lrd_p):
                    ratio = 1 if not np.isinf(lrd_o) else np.inf
                else:
                    ratio = lrd_o / lrd_p
                sum_lrd += ratio
            lof_value = sum_lrd / len(k_neighbors_p)
            return (idx_p, lof_value)

        lof = Parallel(n_jobs=self.n_jobs)(
            delayed(_compute_lof)(idx_p) for idx_p in range(len(D))
        )
        return lof
