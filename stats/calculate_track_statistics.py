from .base import StatsBase

from typing import List, Dict

import numpy as np


class CalculateTrackStatistics(StatsBase):
    def __init__(self):
        """calculate track statics."
        """
    def __call__(self, value: np.ndarray, track_names: List[str]) \
            -> Dict[str, float]:
        """average over each track ignoring sample and song

        Args:
            value (np.ndarray): metrics output
            track_names (List[str]): list of track names

        Returns:
            Dict[str, float]: track name and its value
        """
        lst = list(np.average(value, (0, 1, 2)))
        return {key: value.item() for key, value in zip(track_names, lst)}
