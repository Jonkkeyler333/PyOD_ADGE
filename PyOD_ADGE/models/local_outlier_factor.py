from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted
import scipy.spatial.distance
import numpy as np

class LocalOutlierFactor(BaseEstimator, OutlierMixin):
    def __init__(self,k_neighbors:int=20,metric:str='euclidean',contamination:float=0.1,**metric_params):
        if k_neighbors<1:
            raise ValueError("k_neighbors must be greater than 0")
        if contamination<0 or contamination>0.5:
            raise ValueError("contamination must be between 0 and 0.5")
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.metric_params = metric_params
        self.contamination = contamination
        self._k_distances_cache = {}
                
    def fit(self,X,y=None):
        self._X=X
        if X.shape[0]<self.k_neighbors:
            raise ValueError("The number of samples must be greater than k_neighbors")
        self.metric_c = self._get_distance_metric()
        self._k_distances_cache = self._get_all_k_distances(self._X)
        self.decision_scores_=self._local_outlier_factor(self._X)
        self._is_fitted=True
        return self
    
    def decision_function(self):
        check_is_fitted(self, '_is_fitted')
        if not hasattr(self, '_X'):
            raise ValueError("The model has not been fitted yet.")
        scores=self.outlier_factor_
        self.threshold_ = np.quantile(scores, 1 - self.contamination)
        return np.where(scores>self.threshold_,-1,1)
    
    @property
    def outlier_factor_(self):
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
            rd=np.array(reach_distances)
            srd=np.sum(rd)
            if srd==0:
                srd=np.inf
            lrd.append((idx_p,1/(srd/len(k_neighbors_p))))
        return lrd
            
    def _local_outlier_factor(self,D):
        lof=[]
        lrd = self._local_reachability_density(D)
        for idx_p,p in enumerate(D):
            _,k_neighbors_p=self._k_distances_cache[idx_p]
            lrd_p = lrd[idx_p][1]
            sum_lrd = 0
            for idx_o in k_neighbors_p:
                if np.isinf(lrd_p):
                    ratio = 1 if not np.isinf(lrd[idx_o][1]) else np.inf
                    sum_lrd += ratio
                else:
                    sum_lrd += lrd[idx_o][1]/lrd_p
            lof.append((idx_p,sum_lrd/len(k_neighbors_p)))
        return lof