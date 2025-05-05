from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted
import scipy.spatial.distance
import numpy as np

class LocalOutlierFactor(BaseEstimator, OutlierMixin):
    """Local Outlier Factor (LOF) algorithm for outlier detection.
    This class implements the LOF algorithm, which identifies outliers in a dataset based on the local density of data points.
    """
    def __init__(self,k_neighbors:int=20,metric:str='euclidean',contamination:float=0.1,**metric_params):
        """The constructor for LocalOutlierFactor class.
        :param k_neighbors: The number of neighbors to use for the LOF calculation, defaults to 20
        :type k_neighbors: int, optional
        :param metric: The distance metric to use, defaults to 'euclidean'
        :type metric: str, optional
        :param contamination: The proportion of outliers in the data, defaults to 0.1
        :type contamination: float, optional
        :param metric_params: Additional parameters for the distance metric, defaults to {}
        :type metric_params: dict, optional
        :raises ValueError: If k_neighbors is less than 1 or if contamination is not between 0 and 0.5
        :raises ValueError: If the metric is not recognized
        """
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
        """fit the model to the data.

        :param X: The input data, a 2D array-like structure where each row is a sample and each column is a feature.
        :type X: array-like, shape (n_samples, n_features)
        :param y: This parameter isn't used in this implementation, but it's included for compatibility with scikit-learn's fit method.
        :type y: None
        :raises ValueError: If the number of samples is less than k_neighbors.
        :return: self
        :rtype: LocalOutlierFactor
        """
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
    def outlier_factor_(self)->np.ndarray:
        """Returns the outlier factor for each sample in the dataset.
        The outlier factor is a measure of how much a sample deviates from its neighbors.

        :return: LOF scores for each sample in the dataset.
        :rtype: numpy.ndarray, shape (n_samples,)
        :raises ValueError: If the model has not been fitted yet.
        """
        check_is_fitted(self, '_is_fitted')
        sorted_scores = sorted(self.decision_scores_, key=lambda x: x[0])
        return np.array([score[1] for score in sorted_scores])
    
    def _get_distance_metric(self):
        """Get the distance metric from scipy.spatial.distance based on the metric name.

        :raises ValueError: If the metric is not recognized.
        :return: Distance function from scipy.spatial.distance.
        :rtype: scipy.spatial.distance function
        """
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
        """Calculate the k-distance for a point p in the dataset D.

        :param D: All the data points.
        :type D: np.ndarray
        :param p: The point for which to calculate the k-distance.
        :type p: np.ndarray
        :param idx_p: The index of the point p in the dataset D.
        :type idx_p: int
        :return: The k-distance and the indices of the k-neighbors.
        :rtype: tuple of int and list of int
        """
        distances = []
        for idx,o in enumerate(D):
            if idx == idx_p:
                continue
            distances.append((self.metric_c(o,p),idx))
        distances.sort(key = lambda x: x[0])
        k_distance = distances[self.k_neighbors-1][0]
        k_neigbors = [idx for dist,idx in distances if dist<=k_distance]
        return k_distance, k_neigbors
    
    def _get_all_k_distances(self,D)->dict:
        """Get the k-distance for all points in the dataset D.
        This function calculates the k-distance for each point in the dataset D and caches the results.

        :param D: All the data points.
        :type D: np.ndarray
        :return: A dictionary where the keys are the indices of the points and the values are tuples containing the k-distance and the indices of the k-neighbors.
        :rtype: _dict
        """
        k_distances_cache = {}
        for idx_p,p in enumerate(D):
            if idx_p in k_distances_cache:
                continue
            k_distance, k_neighbors = self._k_distances(D,p,idx_p)
            k_distances_cache[idx_p] = (k_distance, k_neighbors)
        return k_distances_cache
        
    def _reach_dist(self,p,o,k_distance):
        """Calculate the reachability distance between two points p and o.

        :param p: The point for which to calculate the reachability distance.
        :type p: _np.ndarray
        :param o: The other point.
        :type o: _np.ndarray
        :param k_distance: The k-distance of the point o.
        :type k_distance: int
        :return: The reachability distance between p and o.
        :rtype: float
        """
        return max(k_distance,self.metric_c(p,o))
    
    def _local_reachability_density(self,D)->list:
        """Calculate the local reachability density for each point in the dataset D.
        The local reachability density is a measure of how densely populated the local neighborhood of a point is.

        :param D: All the data points.
        :type D: np.ndarray
        :return: A list of tuples where each tuple contains the index of the point and its local reachability density.
        :rtype: list of tuples
        """
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
        """Calculate the Local Outlier Factor (LOF) for each point in the dataset D.

        :param D: All the data points.
        :type D: np.ndarray
        :return: A list of tuples where each tuple contains the index of the point and its LOF score.
        :rtype: _list
        """
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