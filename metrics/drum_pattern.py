from typing import (
    List,
    Dict,
)


import numpy as np

from .base import MetricsBase
from ..reshaper.base import ReshaperBase
from ..stats.base import StatsBase
from .utils import drum_pattern_mask


class DrumPattern(MetricsBase):
    def __init__(
        self,
        n_samples: int,
        songs_per_sample: int,
        measures_per_song: int,
        reshaper: ReshaperBase,
        track_names: List[str],
        postprocess: StatsBase,
        target_track_index: int,
        timesteps_per_measure: int,
    ):
        """Drum pattern metrics

        Args:
            n_samples (int): number of total samples to evaluate
            songs_per_sample (int): number of songs contained one sample
            measures_per_song (int): number of measures contained one song
            reshaper (ReshaperBase): instance of Reshaper
            track_names (List[str]): list of track names
            postprocess (StatsBase): converter that convert self.results into
                dict[track_name: value]
            target_track_index (int): track index used in this metrics
            timesteps_per_measure: (int): timestep per measure
        """
        self.results = np.zeros(
            (n_samples, songs_per_sample, measures_per_song, 1))
        self.songs_per_samples = songs_per_sample
        self.measures_per_songs = measures_per_song
        self.reshaper =reshaper
        self.track_names = track_names
        self.postprocess = postprocess
        self.target_track_index = target_track_index
        self.mask = \
            drum_pattern_mask(timesteps_per_measure)[np.newaxis, np.newaxis, :]

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """convert result into dict.
        first key is the name of metrics(used_pitch_classes).
        second key is the name of the track.

        Returns:
            Dict[str, Dict[str, float]]: metrics value for each track
        """
        return {
            "drum_pattern": self.postprocess(
                self.results, [self.track_names[self.target_track_index]])
        }

    def __call__(self, idx: int, pianoroll: np.ndarray):
        """calculate drum pattern

        Args:
            idx (int): index of the sample
            pianoroll (np.ndarray): pianoroll of n songs
        """
        pianoroll = self.reshaper(pianoroll)

        target_track = pianoroll[..., self.target_track_index]
        n_in_pattern = np.sum(self.mask * np.sum(target_track, axis=3), axis=2)
        n_notes = np.count_nonzero(target_track, axis=(2, 3))
        result = np.where(n_notes > 0, n_in_pattern / n_notes, 0)

        self.results[idx] = result[..., np.newaxis]
