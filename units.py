"""evaluation functions
"""
from abc import (
    abstractmethod,
    ABCMeta,
)
from typing import (
    Union,
    Dict,
    Any,
)

import numpy as np


class EvaluateUnit(ABCMeta):
    @property
    @abstractmethod
    def return_type(self):
        return int  # int, float, or np.ndarray

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        return {"unit_name": 0}

    @abstractmethod
    def __call__(self, pianoroll: np.ndarray) -> Union[int, float, np.ndarray]:
        """run evaluation
        Args:
            pianoroll (np.ndarray): pianoroll of one song

        Returns:
            Union[int, float, np.ndarray]: result
        """
        return

# TODO: implement each methods
