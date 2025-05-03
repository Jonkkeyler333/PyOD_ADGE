from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted
import scipy.spatial.distance
import numpy as np

class LocalOutlierFactor(BaseEstimator, OutlierMixin):
    def __init__(self,k_neighbors:int=20,metric:str='euclidean'):
        self.k_neighbors = k_neighbors
        self.metric = metric
        self._k_distances_cache = {}
                
    def fit(self,X,y=None):
        self._X=X
        self.metric_c = self._get_distance_metric()
        self._k_distances_cache = self._get_all_k_distances(self._X)
        self.decision_scores_=self._local_outlier_factor(self._X)
        self._is_fitted=True
        return self
    
    def decision_function(self,X):
        check_is_fitted(self, '_is_fitted')
        if not hasattr(self, '_X'):
            raise ValueError("The model has not been fitted yet.")
        if X.shape[1] != self._X.shape[1]:
            raise ValueError("The input data must have the same number of features as the training data.")
        if X.shape != self._X.shape or not np.allclose(X, self._X):
            raise NotImplementedError(
                "decision_function currently only supports training data."
            )
        return self.decision_scores_
    
    def _get_distance_metric(self):
            metrics = {
            'euclidean': scipy.spatial.distance.euclidean,
            'manhattan': scipy.spatial.distance.cityblock,
            'cosine': scipy.spatial.distance.cosine,
            'chebyshev': scipy.spatial.distance.chebyshev,
            'minkowski': scipy.spatial.distance.minkowski,
            }
            try:
                return metrics[self.metric]
            except KeyError:
                raise ValueError(f"Unknown metric: {self.metric}.")
    
    def _k_distances(self,D,p,idx_p):
        distances = []
        for idx,o in enumerate(D):
            if idx == idx_p:
                continue
            distances.append((self.metric_c(o,p),idx))
        distances.sort(key = lambda x: x[0])
        k_distance = distances[self.k_neighbors-1][0]
        k_neigbors = [idx for dist,idx in distances if dist<=k_distance]
        return k_distance, k_neigbors
    
    def _get_all_k_distances(self,D):
        k_distances_cache = {}
        for idx_p,p in enumerate(D):
            if idx_p in k_distances_cache:
                continue
            k_distance, k_neighbors = self._k_distances(D,p,idx_p)
            k_distances_cache[idx_p] = (k_distance, k_neighbors)
        return k_distances_cache
        
    def _reach_dist(self,p,o,k_distance):
        return max(k_distance,self.metric_c(p,o))
    
    def _local_reachability_density(self,D):
        lrd = []
        k_distances = self._k_distances_cache
        for idx_p,p in enumerate(D):
            _,k_neighbors_p=k_distances[idx_p]
            reach_distances = []
            for idx_o in k_neighbors_p:
                o = D[idx_o]
                o_k_distance = k_distances[idx_o][0]
                reach_distances.append(self._reach_dist(p,o,o_k_distance))
            lrd.append((idx_p,1/(sum(reach_distances)/len(k_neighbors_p))))
        return lrd
            
    def _local_outlier_factor(self,D):
        lof=[]
        lrd = self._local_reachability_density(D)
        for idx_p,p in enumerate(D):
            _,k_neighbors_p=self._k_distances_cache[idx_p]
            lrd_p = lrd[idx_p][1]
            sum_lrd = 0
            for idx_o in k_neighbors_p:
                sum_lrd += lrd[idx_o][1]/lrd_p
            lof.append((idx_p,sum_lrd/len(k_neighbors_p)))
        return lof