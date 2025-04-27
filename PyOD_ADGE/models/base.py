from abc import ABC,abstractmethod
from typing import Union

class BaseDetector(ABC):
    """_summary_
    """
    def __init__(self,contamination:Union[float,int]=0.1):
        """Constructor for BaseDetector class.

        :param contamination: The amount of outliers expected in the data, defaults to 0.1
        :type contamination: float, optional
        :raises ValueError: _description_
        """
        if isinstance(contamination,(int,float)):
            if not (0<contamination<0.5): 
                raise ValueError("contamination must be between 0 and 0.5") #Si es mayor a 0.5 no tiene sentido
            else:
                self.contamination = contamination
    