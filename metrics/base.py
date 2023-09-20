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

from ..reshaper.base import ReshaperBase
from ..stats.base import StatsBase


class MetricsBase(ABC):
    @abstractmethod
    def __init__(
        self,
        n_samples: int,
        songs_per_sample: int,
        measures_per_song: int,
        reshaper: ReshaperBase,
        track_names: List[str],
        postprocess: StatsBase,
    ):
        """Abstract class for Evaluation metrics

        Args:
            n_samples (int): number of total samples to evaluate
            songs_per_sample (int): number of songs contained one sample
            measures_per_song (int): number of measures contained one song
            reshaper (ReshaperBase): instance of Reshaper
            track_names (List[str]): list of track names
            postprocess (StatsBase): converter that convert self.results into
                dict[track_name: value]
        """
        self.results = np.zeros((n_samples, songs_per_sample))
        self.songs_per_samples = songs_per_sample
        self.measures_per_songs = measures_per_song
        self.reshaper =reshaper
        self.track_names = track_names
        self.postprocess = postprocess

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """average on measure. ignoring sample and song

        Returns:
            Dict[str, Any]: metrics and value
        """
        return {"metrics_name": self.postprocess(self.results)}

    @abstractmethod
    def __call__(self, idx: int, pianoroll: np.ndarray):
        """run evaluation
        Args:
            idx (int): index of the sample
            pianoroll (np.ndarray): pianoroll of n songs.
        """
        return
