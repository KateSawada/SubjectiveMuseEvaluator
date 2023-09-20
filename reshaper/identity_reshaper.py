from typing import List

import numpy as np

from .base import ReshaperBase


class IdentityReshaper(ReshaperBase):
    def __init__(
        self,
        input_shape: List[int],
        songs_per_sample: int,
        measures_per_song: int,
        timesteps_per_measure: int,
        n_pitches: int,
        n_tracks: int,
        hard_threshold: float = 0.5,
        *args,
        **kwargs
    ):
        """this reshaper return input tensor without any reshape process

        Args:
            input_shape (List[int]): shape of input tensor
            songs_per_sample (int): songs per sample
            measures_per_song (int): measures per song
            timesteps_per_measure (int): timesteps per measure
            n_pitches (int): n_pitches
            n_tracks (int): n_tracks
            hard_threshold (float, optional): threshold for float array.
                default to 0.5
        """
        self.input_shape = input_shape
        self.songs_per_samples = songs_per_sample
        self.measures_per_song = measures_per_song
        self.timesteps_per_measure = timesteps_per_measure
        self.n_pitches = n_pitches
        self.n_tracks = n_tracks
        self.hard_threshold = hard_threshold

    def __call__(self, pianoroll: np.ndarray) -> np.ndarray:
        """return input tensor without any reshape process

        Args:
            pianoroll (np.ndarray): pianoroll tensor

        Returns:
            np.ndarray: pianoroll tensor
                shape=(n_songs, n_measures, timesteps, n_pitches, n_tracks)
        """
        self.validation_input_shape(pianoroll)
        return pianoroll
