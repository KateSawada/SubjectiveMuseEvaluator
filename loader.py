"""
loader definitions.
all loader take filename as argument and returns one pianoroll as np.ndarray
"""
from abc import (
    abstractmethod,
    ABC,
)

import torch
import numpy as np


class LoaderBase(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, filename: str) -> np.ndarray:
        """load file and return it as a np.ndarray

        Args:
            filename (str): filename

        Returns:
            np.ndarray: pianoroll
        """
        return np.load(filename)


class NDarrayFromPt(LoaderBase):
    def __init__(self) -> None:
        """this class load pt file that contains only one np.ndarray(pianoroll)
        """
        pass

    def __call__(self, filename: str) -> np.ndarray:
        """load file and return it as a np.ndarray

        Args:
            filename (str): filename

        Returns:
            np.ndarray: pianoroll
        """
        return torch.load(filename)


class NDarrayFromNoiseListPt(LoaderBase):
    def __init__(self) -> None:
        """this class load pt file that was exported by default DiffRoll
        """
        pass

    def __call__(self, filename: str) -> np.ndarray:
        """load file and return it as a np.ndarray

        Args:
            filename (str): filename

        Returns:
            np.ndarray: pianoroll
        """
        return torch.load(filename)[-1][0]


class NDarrayFromNpy(LoaderBase):
    def __init__(self) -> None:
        """this class load npy file that contains one sample
        """
        pass

    def __call__(self, filename: str) -> np.ndarray:
        """load file and return it as a np.ndarray

        Args:
            filename (str): filename

        Returns:
            np.ndarray: pianoroll
        """
        return np.load(filename)
