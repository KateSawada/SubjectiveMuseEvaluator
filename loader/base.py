from abc import (
    abstractmethod,
    ABC,
)

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
