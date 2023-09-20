from abc import (
    abstractmethod,
    ABC,
)
from typing import (
    List,
    Dict,
    Any,
)

import numpy as np


class StatsBase(ABC):
    def __init__(self, track_names: List[str]):
        """calculate track statics."

        Args:
            track_names (List[str]): list of track names
        """
        self.track_names = track_names

    @abstractmethod
    def __call__(self, value: np.ndarray) -> Dict[str, Any]:
        """average over each track ignoring sample and song

        Args:
            value (np.ndarray): metrics output

        Returns:
            Dict[str, Any]: track name and its value
        """
        lst = list(np.average(value, axis=(0, 1, 2)))
        return {key: value for key, value in zip(self.track_names, lst)}
