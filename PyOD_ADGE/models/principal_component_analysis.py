"""
Principal Component Analysis (PCA) for anomaly detection.

"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from .base import BaseDetector
from numpy import percentile
from sklearn.utils.validation import check_array,check_is_fitted

class PCA(BaseDetector):
    """Principal Component Analysis (PCA)
    This class implements PCA for outlier detection. It is a wrapper around the sklearn PCA implementation.
    It computes the PCA components and explained variance, and uses them to compute outlier scores.
    The outlier scores are computed based on the Mahalanobis distance of the data points to the PCA components.
    
    :param n_components: The number of components to use for PCA. If None, all components are used.
    :type n_components: int, optional
    :param n_selected_components: The number of components to use for PCA. If None, all components are used.
    :type n_selected_components: int, optional
    :param contamination: The proportion of outliers in the data set. This is used to determine the threshold for outlier detection.
    :type contamination: float, optional
    
    """
    def __init__(self, n_components=None, n_selected_components=None,
                 contamination:float=0.1, copy=True, whiten=False, svd_solver='auto',
                 tol=0.0, iterated_power='auto', random_state=None,
                 weighted=True, standardization=True):
        
        if svd_solver not in ['auto', 'full','covariance_eigh', 'arpack', 'randomized']:
            raise ValueError("El svd_solver debe ser uno de los siguientes : ['auto', 'full', 'arpack', 'randomized']")
        if n_components is None and n_selected_components is None:
            raise ValueError("n_components y n_selected_components no pueden ser ambos None")
        if n_components is not None and n_components <= 0:
            raise ValueError("n_components debe ser mayor que 0")
        if n_selected_components is not None and n_selected_components <= 0:
            raise ValueError("n_selected_components debe ser mayor que 0")
        if contamination < 0 or contamination > 0.5:
            raise ValueError("contamination debe ser entre 0 y 0.5")
        if n_components is None and n_selected_components is None:
            raise ValueError("n_components y n_selected_components no pueden ser ambos None")
        
        if n_components is None:
            n_components = n_selected_components  
        if n_selected_components is None:
            n_selected_components = n_components
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.contamination = contamination
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.weighted = weighted
        self.standardization = standardization
        

    def fit(self, x, y=None):
        """Fit the model to the data.
        This method fits the PCA model to the input data and computes the outlier scores.
        :param x: The input data, a 2D array-like structure where each row is a sample and each column is a feature.
        :type x: array-like, shape (n_samples, n_features)
        :param y: This parameter isn't used in this implementation, but it's included for compatibility with scikit-learn's fit method.
        :type y: None , optional
        :return: self, the fitted model
        """
        X = check_array(x)
        self._X = X

        if self.standardization:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = None

        self.detector = sklearnPCA(
            n_components=self.n_components,
            copy=self.copy,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            random_state=self.random_state
        )
        self.detector.fit(X)
        
        if self.n_selected_components is None:
            self.n_selected_components = self.n_components
        if self.n_components is None:
            self.n_components = self.n_selected_components

        if self.n_selected_components is not None:
            self.n_components = self.n_selected_components
        else:
            self.n_selected_components = self.n_components
        if self.n_components is not None:
            if isinstance(self.n_components, float):
                self.n_selected_components = self.detector.components_.shape[0]
            else:
                self.n_selected_components = self.n_components
            self.components_ = self.detector.components_[:self.n_components]
            self.explained_variance_ = self.detector.explained_variance_[:self.n_components]
            self.explained_variance_ratio_ = self.detector.explained_variance_ratio_[:self.n_components]
        else:
            self.components_ = self.detector.components_
            self.explained_variance_ = self.detector.explained_variance_
            self.explained_variance_ratio_ = self.detector.explained_variance_ratio_
        
        self.mean_ = self.detector.mean_
        self.singular_values_ = self.detector.singular_values_
        self.noise_variance_ = self.detector.noise_variance_

        if self.weighted:
            self.w_components_ = self.explained_variance_ratio_
        else:
            self.w_components_ = np.ones(self.n_components)

        self.selected_components_ = self.components_[-self.n_selected_components:, :]
        self.selected_w_components_ = self.w_components_[-self.n_selected_components:]

        distancias = cdist(X, self.selected_components_)
        self.decision_scores_ = np.sum(distancias / self.selected_w_components_, axis=1).ravel()

        self._process_decision_scores()
        return self

    
    def decision_function(self, X):
        """Compute the decision function for the fitted model.
        This method calculates the outlier scores for the input data based on the fitted model.
        The decision function is the Mahalanobis distance of the data points to the PCA components.
        :param x: The input data, a 2D array-like structure where each row is a sample and each column is a feature.
        :type x: array-like, shape (n_samples, n_features)
        :return: The outlier scores of the input samples. Shape (n_samples,)
        """
        check_is_fitted(self, ['components_', 'w_components_'])
        X = check_array(X)
        if self.standardization:
            X = self.scaler.transform(X)
        # Calcular las distancias y ponderarlas
        distancias = cdist(X, self.selected_components_)
        decision_scores = np.sum(distancias / self.selected_w_components_, axis=1).ravel()
        return decision_scores
    
    def _process_decision_scores(self):
        """Process the decision scores to determine the outlier labels and threshold."""

        if isinstance(self.contamination, (float, int)):
            self.threshold_ = percentile(self.decision_scores_,
                                         100 * (1 - self.contamination))
            self.labels_ = (self.decision_scores_ > self.threshold_).astype(
                'int').ravel()
        else:
            self.labels_ = self.contamination.eval(self.decision_scores_)
            self.threshold_ = self.contamination.thresh_
            if not self.threshold_:
                self.threshold_ = np.sum(self.labels_) / len(self.labels_)

        self._mu = np.mean(self.decision_scores_)
        self._sigma = np.std(self.decision_scores_)
        return self
    
    def fit_transform(self, X,y=None):
        """Fit the model to the data and transform it."""
        X = check_array(X)
        self.fit(X)  

        if self.standardization and self.scaler is not None:
            X = self.scaler.transform(X)

        self.X_transformed = self.detector.transform(X)[:, :self.n_components]
        return self.X_transformed
    






