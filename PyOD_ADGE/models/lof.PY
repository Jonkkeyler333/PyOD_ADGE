from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from .base import BaseDetector
from .local_outlier_factor import LocalOutlierFactor
from sklearn.neighbors import LocalOutlierFactor as LOF_Sklearn
import numpy as np

class LOF(BaseDetector):
    """Local Outlier Factor (LOF) algorithm for outlier detection.
    This class implements the LOF algorithm, which identifies outliers in a dataset based on the local density of data points.
    It is a wrapper around the own implementation or scikit learn LOF.
    """
    def __init__(self,estimator:str,n_neighbors:int = 20,contamination:float = 0.1,metric = 'euclidean', **metric_params)->None:
        super(LOF,self).__init__(contamination)
        if estimator == 'sklearn':
            estimator_cls = LOF_Sklearn
        elif estimator == 'clasic':
            estimator_cls = LocalOutlierFactor
        else:
            raise ValueError("estimator debe ser 'sklearn' o 'clasic'")
        self.n_neighbors = n_neighbors
        self.is_sklearn = True if estimator_cls == LOF_Sklearn else False
        self.metric = metric
        self.metric_params = metric_params
        self.estimator = estimator_cls
    
    def fit(self,x,y=None)->'LOF':
        """Fit the model to the data.
        This method fits the LOF model to the input data and computes the outlier scores.

        :param x: The input data, a 2D array-like structure where each row is a sample and each column is a feature.
        :type x: array-like, shape (n_samples, n_features)
        :param y: This parameter isn't used in this implementation, but it's included for compatibility with scikit-learn's fit method.
        :type y: None , optional
        :return: self, the fitted model
        :rtype: LOF
        """
        X=check_array(x)
        self.detector = self.estimator(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            metric=self.metric,
            **self.metric_params)
        self.detector.fit(X,y)
        if self.is_sklearn:
            self.decision_scores_ = self.detector.negative_outlier_factor_
        self.decision_scores_ = self.detector.outlier_factor_
        return self
    
    @property
    def outlier_scores_(self)->np.ndarray:
        """""Get the outlier scores of the fitted model.
        This property returns the outlier scores of the fitted model. The scores are computed based on the local density of data points.
        :return: The outlier scores of the fitted model. Shape (n_samples,)
        :rtype: np.ndarray
        """
        check_is_fitted(self,'detector')
        return self.decision_scores_
    
    def decision_function(self)->np.ndarray:
        """Predict the outlier score of the input samples.
        This method computes the outlier scores for the input samples based on the fitted model.
        :return: The outlier scores of the input samples. Shape (n_samples,)
        :rtype: np.ndarray
        """
        check_is_fitted(self,'detector')
        if self.is_sklearn:
            return self.detector.decision_function()
        return self.detector.decision_function()