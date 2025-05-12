from .base import BaseDetector
import numpy as np
from .lof import LOF
from sklearn.utils.validation import check_array,check_is_fitted
from joblib import Parallel,delayed
from sklearn.base import clone

class FeatureBagging(BaseDetector):
    def __init__(self,base_estimator:str='classic',
                 n_estimators:int=10,
                 contamination:float=0.1,
                 combine:str='breadth',
                 n_jobs:int=2,
                 random_state:int=None,
                 n_neighbors:int = 20,
                 metric = 'euclidean',
                 **metric_params)->None:
        """The FeatureBagging class is a feature bagging ensemble method for outlier detection.
        It is a subclass of the BaseDetector class and implements the fit and decision_function methods.
        The class uses the LOF (Local Outlier Factor) algorithm as the base estimator for outlier detection.

        :param base_estimator: the version of the LOF algorithm, defaults to 'classic'
        :type base_estimator: str, optional
        :param n_estimators: the number of base estimators to use, defaults to 10
        :type n_estimators: int, optional
        :param contamination: the proportion of outliers in the data set, defaults to 0.1
        :type contamination: float, optional
        :param combine: the method to combine the results of the base estimators, defaults to 'breadth'
        :type combine: str, optional
        :param n_jobs: the number of jobs to run in parallel, defaults to 2
        :type n_jobs: int, optional
        :param random_state: the random seed for reproducibility, defaults to None
        :type random_state: int, optional
        :param n_neighbors: the number of neighbors to use for the LOF algorithm, defaults to 20
        :type n_neighbors: int, optional
        :param metric: the distance metric to use for the LOF algorithm, defaults to 'euclidean'
        :type metric: str, optional
        :raises ValueError: _base_estimator debe ser 'classic' o 'sklearn'
        :raises ValueError: _el combine debe ser 'breadth' o 'Cumulative'
        :raises ValueError: _n_estimators debe ser mayor que 1
        :raises ValueError: _contamination debe ser entre 0 y 0.5
        """
        if combine not in ['breadth','cumulative']:
            raise ValueError("el combine debe ser 'breadth' o 'Cumulative'")
        if base_estimator not in ['classic','sklearn']:
            raise ValueError("el base_estimator debe ser 'classic' o 'sklearn'")
        if n_estimators<=1:
            raise ValueError("n_estimators debe ser mayor que 1")
        if contamination<0 or contamination>0.5:
            raise ValueError("contamination debe ser entre 0 y 0.5")
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.combine = combine
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_params = metric_params
    
    def fit(self,X,y=None)->'FeatureBagging':
        """Train the model using the provided data.
        This method fits the model to the data by training multiple base estimators on different subsets of features.

        :param X: the input data to fit the model, shape (n_samples, n_features)
        :type X: np.ndarray
        :param y: the target labels useless lof is unsupervised, defaults to None
        :type y: _type_, optional
        :return: the fitted model
        :rtype: FeatureBagging
        """
        X=check_array(X)
        self.d=X.shape[0]
        ss = np.random.SeedSequence(self.random_state)
        rngs = ss.spawn(self.n_estimators)
        results=Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_subset)(X,LOF(
            estimator=self.base_estimator,
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            metric=self.metric,
            **self.metric_params),rngs[i]) for i in range(self.n_estimators))
        self.subsets_,self.estimators_scores_=zip(*results)
        self._is_fitted=True
        return self
    
    def decision_function(self)->np.ndarray:
        """Compute the decision function for the fitted model.
        This method calculates the outlier scores for the input data based on the fitted model.

        :return: the outlier scores for the input data, shape (n_samples,)
        :rtype: np.ndarray
        """
        check_is_fitted(self, '_is_fitted')
        if self.combine == 'cumulative':
            scores=self._cumulative(self.estimators_scores_)
            self.threshold_ = np.quantile(scores, 1 - self.contamination)
            return np.where(scores>self.threshold_,-1,1)
        else: 
            scores,idx_scores=self._breadth_first(self.estimators_scores_)
        self.threshold_ = np.quantile(scores, 1 - self.contamination)
        idx_sort=np.argsort(idx_scores)
        scores=scores[idx_sort]
        return np.where(scores>self.threshold_,-1,1)
    
    @property
    def outlier_factor_(self)->np.ndarray:
        """Compute the outlier factor for the fitted model.
        This method calculates the outlier scores for the input data based on the fitted model.

        :return: the outlier scores for the input data, shape (n_samples,)
        :rtype: np.ndarray
        """
        check_is_fitted(self, '_is_fitted')
        ASV=self.estimators_scores_
        if self.combine == 'cumulative':
            ASV_final=self._cumulative(ASV)
            return ASV_final
        ASV_final,idx_ASV_final=self._breadth_first(ASV)
        idx_sort=np.argsort(idx_ASV_final)
        scores=ASV_final[idx_sort]
        return scores

    def _bagging_features_idx(self,d:int,rng)->np.ndarray:
        """this function generates a random subset of features for bagging.
        It selects a random number of features from the input data and returns their indices.

        :param d: the number of features in the input data
        :type d: int
        :param rng: the random number generator
        :type rng: np.random.Generator
        :return: the indices of the selected features for bagging
        :rtype: np.ndarray
        """
        low=d//2
        n_features=rng.integers(low,d)
        return rng.choice(d,n_features,replace=False)
    
    def _train_subset(self,X,estimator,rng_seq)->tuple:
        """Train a base estimator on a random subset of features.
        This method selects a random subset of features from the input data and fits the base estimator to that subset.

        :param X: the input data to fit the model, shape (n_samples, n_features)
        :type X: np.ndarray
        :param estimator: the base estimator to fit to the subset of features
        :type estimator: BaseDetector
        :param rng_seq: the random number generator seed
        :type rng_seq: 
        :return: the indices of the selected features and the outlier scores from the base estimator
        :rtype: tuple
        """
        rng = np.random.default_rng(rng_seq)
        idx = self._bagging_features_idx(X.shape[1], rng)
        X_sub = X[:, idx]
        estimator.fit(X_sub)
        return idx, estimator.outlier_scores_
    
    def _breadth_first(self,ASV)->tuple:
        """method to combine the results of the base estimators using breadth-first search.
        It sorts the outlier scores and returns the sorted scores and their corresponding indices.

        :param ASV: the outlier scores from the base estimators, shape (n_estimators, n_samples)
        :type ASV: np.ndarray
        :raises ValueError: el numero de estimadores no coincide con el numero de ASV
        :return: the sorted outlier scores and their corresponding indices
        :rtype: tuple
        """
        ASV=np.stack(ASV)
        if not ASV.shape[0] == self.n_estimators:
            raise ValueError("el numero de estimadores no coincide con el numero de ASV")
        if self.base_estimator == 'sklearn':
            sort_idx_ASV=np.argsort(-ASV,axis=1)
            sort_ASV=np.take_along_axis(ASV,sort_idx_ASV,axis=1)
        else:
            sort_idx_ASV=np.argsort(ASV,axis=1)
            sort_ASV=np.take_along_axis(ASV,sort_idx_ASV,axis=1)
        ASV_final=[]
        idx_ASV_final=[]
        for i in range(self.d):
            for j in range(self.n_estimators):
                elem=sort_ASV[j,i]
                if sort_idx_ASV[j,i] in idx_ASV_final:
                    continue
                ASV_final.append(elem)
                idx_ASV_final.append(sort_idx_ASV[j,i])
        IndFINAL = np.array(idx_ASV_final, dtype=int)
        ASV_final  = np.array(ASV_final, dtype=float)
        return ASV_final,IndFINAL
    
    def _cumulative(self,ASV)->np.ndarray:
        """method to combine the results of the base estimators using cumulative sum.
        It sums the outlier scores from all base estimators and returns the final outlier scores.

        :param ASV: the outlier scores from the base estimators, shape (n_estimators, n_samples)
        :type ASV: np.ndarray
        :raises ValueError: el numero de estimadores no coincide con el numero de ASV 
        :return: the final outlier scores
        :rtype: np.ndarray
        """
        ASV=np.stack(ASV)
        if not ASV.shape[0] == self.n_estimators:
            raise ValueError("el numero de estimadores no coincide con el numero de ASV")
        ASV_final=ASV.sum(axis=0)
        return ASV_final