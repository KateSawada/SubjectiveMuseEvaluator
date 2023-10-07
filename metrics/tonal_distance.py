from typing import (
    List,
    Dict,
)


import numpy as np

from .utils import to_chroma, create_tonal_matrix
from .base import MetricsBase
from ..reshaper.base import ReshaperBase
from ..stats.base import StatsBase


class TonalDistance(MetricsBase):
    def __init__(
        self,
        n_samples: int,
        songs_per_sample: int,
        measures_per_song: int,
        reshaper: ReshaperBase,
        track_names: List[str],
        postprocess: StatsBase,
        timesteps_per_measure: int,
        beat_per_measure: int,
    ):
        """tonal distance metrics

        Args:
            n_samples (int): number of total samples to evaluate
            songs_per_sample (int): number of songs contained one sample
            measures_per_song (int): number of measures contained one song
            reshaper (ReshaperBase): instance of Reshaper
            track_names (List[str]): list of track names
            postprocess (StatsBase): converter that convert self.results into
                dict[track_name: value]
            timesteps_per_measure (int): timestep per measure
            beat_per_measure (int): beat per measure
        """
        # each measure has 2x2 matrix(n_tracks x n_tracks)
        self.results = np.zeros((
            n_samples,
            songs_per_sample,
            measures_per_song,
            len(track_names),
            len(track_names)))
        self.songs_per_samples = songs_per_sample
        self.measures_per_songs = measures_per_song
        self.reshaper =reshaper
        self.track_names = track_names
        self.postprocess = postprocess
        self.timesteps_per_beat = timesteps_per_measure // beat_per_measure

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """convert result into dict.
        first key is the name of metrics(used_pitch_classes).
        second key is the name of the track.

        Returns:
            Dict[str, Dict[str, float]]: metrics value for each track
        """
        track_pairs = []
        results_1d = np.zeros(
            [
                *self.results.shape[0:3],
                (len(self.track_names) * (len(self.track_names) - 1)) // 2,
            ]
        )
        idx = 0
        for i in range(len(self.track_names)):
            for j in range(i + 1, len(self.track_names)):
                track_pairs.append(
                    f"{self.track_names[i]}-{self.track_names[j]}")
                results_1d[..., idx] = self.results[..., i, j]
                idx += 1
        return {
            "tonal_distance": self.postprocess(
                results_1d, track_pairs)
        }

    def __call__(self, idx: int, pianoroll: np.ndarray):
        """calculate tonal distance

        Args:
            idx (int): index of the sample
            pianoroll (np.ndarray): pianoroll of n songs
        """
        pianoroll = self.reshaper(pianoroll)
        chroma = to_chroma(pianoroll)

        mapped = self.to_tonal_space(chroma)
        expanded1 = np.expand_dims(mapped, -1)
        expanded2 = np.expand_dims(mapped, -2)
        tonal_dist = np.linalg.norm(expanded1 - expanded2, axis=0)
        result = np.mean(tonal_dist, 0)

        self.results[idx] = result

    def to_tonal_space(self, pianoroll: np.ndarray) -> np.ndarray:
        """Return the tensor in tonal space where chroma features are normalized
        per beat.

        Args:
            pianoroll (np.ndarray): chroma features array.
                shape=(n_songs, n_measures, timesteps, 12, n_tracks)

        Returns:
            np.ndarray: tonal vector
            shape=(6, -1, tracks)
        """
        tonal_matrix = create_tonal_matrix()
        beat_chroma = np.sum(np.reshape(
            pianoroll,
            (-1, self.timesteps_per_beat, 12, pianoroll.shape[4])), 1)
        # >>> beat_chroma.shape  # (measures, 12, n_tracks)
        beat_chroma = beat_chroma / np.sum(beat_chroma, 1, keepdims=True)
        beat_chroma = np.nan_to_num(beat_chroma)  # remove nan
        reshaped = np.reshape(beat_chroma.transpose(1, 0, 2), (12, -1))
        return np.reshape(
            np.matmul(tonal_matrix, reshaped),
            (6, -1, pianoroll.shape[4]))
