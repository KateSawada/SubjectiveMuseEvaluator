from abc import (
    abstractmethod,
    ABC,
)

import numpy as np


class ReshaperBase(ABC):
    @abstractmethod
    def __call__(self, pianoroll: np.ndarray) -> np.ndarray:
        return pianoroll


class Reshaper(ReshaperBase):
    def __init__(
        self,
        songs_per_sample: int,
        n_tracks: int,
        measures_per_song: int,
        timesteps_per_measure: int,
        n_pitches: int,
        hard_threshold: float = 0.5,
    ):
        """reshaper

        Args:
            songs_per_sample (int): songs per sample
            n_tracks (int): number of tracks
            measures_per_song (int): measures per song
            timesteps_per_measure (int): timesteps per measure
            n_pitches (int): number of pitches
            hard_threshold (float, optional): threshold. default is 0.5
        """
        self.songs_per_sample = songs_per_sample
        self.n_tracks = n_tracks
        self.measures_per_song = measures_per_song
        self.timesteps_per_measure = timesteps_per_measure
        self.n_pitches = n_pitches
        self.hard_threshold = hard_threshold

    def __call__(self, pianoroll: np.ndarray):
        pianoroll = pianoroll > self.hard_threshold
        return pianoroll.reshape(
            self.songs_per_sample,
            self.n_tracks,
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
        return super.__call__(pianoroll)[:, 0, :, :]
