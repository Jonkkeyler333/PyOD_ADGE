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
        if combine not in ['breadth','Cumulative']:
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
    
    def fit(self,X,y=None):
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
    
    def decision_function(self,X):
        check_is_fitted(self, '_is_fitted')
        scores,idx_scores=self._breadth_first(self.estimators_scores_)
        self.threshold_ = np.quantile(scores, 1 - self.contamination)
        idx_sort=np.argsort(idx_scores)
        scores=scores[idx_sort]
        return np.where(scores>self.threshold_,-1,1)
    
    @property
    def outlier_factor_(self):
        check_is_fitted(self, '_is_fitted')
        ASV=self.estimators_scores_
        if self.combine == 'breadth':
            ASV_final,idx_ASV_final=self._breadth_first(ASV)
        else:
            ASV_final,idx_ASV_final=self._cumulative(ASV)
        idx_sort=np.argsort(idx_ASV_final)
        scores=ASV_final[idx_sort]
        return scores
    
    def get(self):
        """Funcion de prueba , despues se eliminara
        """
        return self.subsets_,self.estimators_scores_

    def _bagging_features_idx(self,d,rng)->np.ndarray:
        low=d//2
        n_features=rng.integers(low,d)
        return rng.choice(d,n_features,replace=False)
    
    def _train_subset(self,X,estimator,rng_seq)->np.ndarray:
            rng = np.random.default_rng(rng_seq)
            idx = self._bagging_features_idx(X.shape[1], rng)
            X_sub = X[:, idx]
            estimator.fit(X_sub)
            return idx, estimator.outlier_scores_
    
    def _breadth_first(self,ASV):
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
    
    def _cumulative(self,ASV):
        pass