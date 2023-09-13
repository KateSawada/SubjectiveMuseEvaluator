"""evaluation functions
"""
from abc import (
    abstractmethod,
    ABC,
)
from typing import (
    List,
    Dict,
    Any,
    Callable,
)

import numpy as np


class EvaluateUnit(ABC):
    @abstractmethod
    def __init__(
        self,
        n_samples: int,
        n_pianoroll_per_samples: int,
        reshaper: Callable[[np.ndarray], np.ndarray],
        track_names: List[str],
    ):
        """Abstract class for Evaluation Unit

        Args:
            n_samples (int): count of total samples
            n_pianoroll_per_samples (int): pianoroll per samples
            reshaper (Callable[[np.ndarray], np.ndarray]): reshape function.
                it convert pianoroll shape into (song, track, timestep, pitch)
            track_names (List[str]): list of track name
        """
        self.results = np.zeros(n_samples)
        self.n_pianoroll_per_samples = n_pianoroll_per_samples
        self.reshaper =reshaper
        self.track_names = track_names

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        return {"unit_name": 0}

    @abstractmethod
    def __call__(self, idx: int, pianoroll: np.ndarray):
        """run evaluation
        Args:
            idx (int): index of the sample
            pianoroll (np.ndarray): pianoroll of n songs.
        """
        return


class EmptyBars(EvaluateUnit):
    def __init__(
        self,
        n_samples: int,
        n_pianoroll_per_samples: int,
        reshaper: Callable[[np.ndarray], np.ndarray],
        track_names: List[str],
    ):
        """Empty bars ratio
        Args:
            n_samples (int): count of total samples
            n_pianoroll_per_samples (int): pianoroll per samples
            reshaper (Callable[[np.ndarray], np.ndarray]): reshape function.
                it convert pianoroll shape into (song, track, timestep, pitch)
            track_names (List[str]): list of track name
        """
        self.results = np.zeros((n_samples, len(track_names)))
        self.n_pianoroll_per_samples = n_pianoroll_per_samples
        self.reshaper =reshaper
        self.track_names = track_names

    def __call__(self, idx: int, pianoroll: np.ndarray) -> None:
        """calculate empty bars ratio

        Args:
            idx (int): index of the sample
            pianoroll (np.ndarray): pianoroll of n songs.
        """
        pianoroll = self.reshaper(pianoroll)

        result = np.all(pianoroll == 0, axis=(2, 3)).astype(np.float32)
        self.results[
            idx * self.n_pianoroll_per_samples:
                (idx + 1) * self.n_pianoroll_per_samples] = result

    def to_dict(self):
        results = {}
        for i in range(len(self.track_names)):
            results[self.track_names[i]] = np.average(self.results[:, i]).item()
        return {"empty_bars": results}


# TODO: implement each methods
