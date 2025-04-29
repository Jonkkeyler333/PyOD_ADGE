from abc import ABC,abstractmethod
from typing import Union

class BaseDetector(ABC):
    """_summary_
    """
    @abstractmethod
    def __init__(self,contamination:Union[float,int]=0.1):
        """Constructor for BaseDetector class.

        :param contamination: The amount of outliers expected in the data, defaults to 0.1
        :type contamination: float, optional
        :raises ValueError: _description_
        """
        if isinstance(contamination,(int,float)):
            if not 0<contamination<0.5:
                raise ValueError("contamination must be between 0 and 0.5") #Si es mayor a 0.5 no tiene sentido
            self.contamination = contamination
        else:
            raise ValueError("contamination must be a float or int")
 
    @abstractmethod
    def fit(self,x,y=None):
        """Fit a model detector to the data.

        :param x: The input sample to fit.Shape (n_samples, n_features)
        :type x: numpy.ndarray,list,pandas.Dataframe,pandas.Series
        :param y: The ground truth label, defaults to None if the model is unsupervised
        :type y: numpy.ndarray,list,pandas.Dataframe,pandas.Series, optional
        :return: self, the fitted model
        """
        pass
    
    @abstractmethod
    def decision_function(self,x):
        """Predict the outlier score of the input samples.

        :param x: The input sample to predict. Shape (n_samples, n_features)
        :type x: numpy.ndarray,list,pandas.Dataframe,pandas.Series
        :return: The outlier score of the input samples. Shape (n_samples,)
        """
        pass