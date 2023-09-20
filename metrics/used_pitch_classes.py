from typing import (
    List,
    Dict,
)


import numpy as np

from .utils import to_chroma
from .base import MetricsBase
from ..reshaper.base import ReshaperBase
from ..stats.base import StatsBase


class UsedPitchClasses(MetricsBase):
    def __init__(
        self,
        n_samples: int,
        songs_per_sample: int,
        measures_per_song: int,
        reshaper: ReshaperBase,
        track_names: List[str],
        postprocess: StatsBase,
    ):
        """Used pitch classes metrics

        Args:
            n_samples (int): number of total samples to evaluate
            songs_per_sample (int): number of songs contained one sample
            measures_per_song (int): number of measures contained one song
            reshaper (ReshaperBase): instance of Reshaper
            track_names (List[str]): list of track names
            postprocess (StatsBase): converter that convert self.results into
                dict[track_name: value]
        """
        self.results = np.zeros(
            (n_samples, songs_per_sample, measures_per_song, len(track_names)))
        self.songs_per_samples = songs_per_sample
        self.measures_per_songs = measures_per_song
        self.reshaper =reshaper
        self.track_names = track_names
        self.postprocess = postprocess

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """convert result into dict.
        first key is the name of metrics(used_pitch_classes).
        second key is the name of the track.

        Returns:
            Dict[str, Dict[str, float]]: metrics value for each track
        """
        return {
            "used_pitch_classes": self.postprocess(
                self.results, self.track_names)
        }

    def __call__(self, idx: int, pianoroll: np.ndarray):
        """calculate used pitch classes

        Args:
            idx (int): index of the sample
            pianoroll (np.ndarray): pianoroll of n songs
        """
        pianoroll = self.reshaper(pianoroll)
        chroma = to_chroma(pianoroll)

        result = np.count_nonzero(np.sum(chroma, 2), 2)
        self.results[idx] = result
