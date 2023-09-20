from abc import (
    abstractmethod,
    ABC,
)
from typing import List

import numpy as np


class ReshaperBase(ABC):
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
        **kwargs,
    ):
        """reshaper base class

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

    def validation_input_shape(self, input_tensor: np.ndarray) -> None:
        """validate input tensor shape

        Args:
            input_tensor (np.ndarray): tensor

        Raises:
            ValueError:
                when input tensor size is different from self.input_shape.
        """
        if (self.input_shape != input_tensor.shape):
            raise ValueError(
                f"Input shape {input_tensor.shape} is wrong. "
                f"This reshaper takes {self.input_shape}."
            )

    @abstractmethod
    def __call__(self, pianoroll: np.ndarray) -> np.ndarray:
        """reshape input tensor

        Args:
            pianoroll (np.ndarray): pianoroll tensor

        Returns:
            np.ndarray: reshaped tensor
        """
        return pianoroll
