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
    def __init__(self,estimator,n_neighbors = 20,contamination = 0.1,metric = 'euclidean', **metric_params)->None:
        super(LOF,self).__init__(contamination)
        if not isinstance(estimator,LocalOutlierFactor) or not isinstance(estimator,LOF_Sklearn):
            raise ValueError('estimator must be an instance of LocalOutlierFactor or LOF_Sklearn')
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_params = metric_params
        self.estimator = estimator
    
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
        self.detector = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            metric=self.metric,
            **self.metric_params)
        self.detector.fit(X,y)
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
        return self.detector.decision_function()