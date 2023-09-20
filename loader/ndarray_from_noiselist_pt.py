import numpy as np
import torch

from .base import LoaderBase


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
