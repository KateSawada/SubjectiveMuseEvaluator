
import numpy as np

from .base import LoaderBase


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
