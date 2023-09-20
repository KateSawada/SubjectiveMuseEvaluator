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
        **kwargs
    ):
        """initialize reshaper

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

    @abstractmethod
    def __call__(self, pianoroll: np.ndarray) -> np.ndarray:
        return pianoroll


class Reshaper(ReshaperBase):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """reshaper

        Args:
            input_shape (List[int]): shape of input tensor
            songs_per_sample (int): songs per sample
            measures_per_song (int): measures per song
            timesteps_per_measure (int): timesteps per measure
            n_pitches (int): number of pitches
            n_tracks (int): number of tracks
            hard_threshold (float, optional): threshold. default is 0.5
        """
        super().__init__(*args, **kwargs)

    def __call__(self, pianoroll: np.ndarray):
        pianoroll = pianoroll > self.hard_threshold
        return pianoroll.reshape(
            self.songs_per_sample,
            self.n_tracks,
            -1,  # to suit LPDTrackReshaper
            self.measures_per_song * self.timesteps_per_measure,
            self.n_pitches,
        )


class DrumReshaper(Reshaper):
    def __init__(self, drum_index: int, *args, **kwargs):
        """reshaper for drum track evaluation

        Args:
            songs_per_sample (int): songs per sample
            n_tracks (int): number of tracks
            measures_per_song (int): measures per song
            timesteps_per_measure (int): timesteps per measure
            n_pitches (int): number of pitches
            hard_threshold (float, optional): threshold. default is 0.5
            drum_index (int): index of drum tracks
        """
        super().__init__(*args, **kwargs)
        self.drum_index = drum_index

    def __call__(self, pianoroll: np.ndarray):
        return super().__call__(pianoroll)[:, self.drum_index, :, :]


class LPDTrackReshaper(Reshaper):
    def __init__(self, used_index: List[int], *args, **kwargs):
        """reshaper for drum track evaluation

        Args:
            songs_per_sample (int): songs per sample
            n_tracks (int): number of tracks
            measures_per_song (int): measures per song
            timesteps_per_measure (int): timesteps per measure
            n_pitches (int): number of pitches
            hard_threshold (float, optional): threshold. default is 0.5
            used_index (int): index of track
        """
        super().__init__(*args, **kwargs)
        self.used_index = used_index

    def __call__(self, pianoroll: np.ndarray):
        return super().__call__(pianoroll)[:, self.used_index, :, :]
