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


def _to_dict_average_over_track(
    track_names: List[str],
    results: np.ndarray,
) -> Dict[str, float]:
    results_dict = {}
    for i in range(len(track_names)):
        results_dict[track_names[i]] = np.average(results[:, i]).item()
    return results_dict


def _add_measure_axis(
    pianoroll: np.ndarray,
    measures_per_song: int,
) -> np.ndarray:
    """add measure axis.
    shape: (songs, track, timestep, pitch)
        -> (songs, track, measure, timestep, pitch)

    Args:
        pianoroll (np.ndarray): original pianoroll.
            shape: (songs, timestep, pitch)
        measures_per_song (int): measures per song

    Returns:
        np.ndarray: reshaped pianoroll.
            shape: (songs, track, measure, timestep, pitch)
    """
    return np.reshape(
        pianoroll,
        (pianoroll.shape[0], pianoroll.shape[1],
         measures_per_song, -1, pianoroll.shape[3])
    )


class EmptyBars(EvaluateUnit):
    def __init__(
        self,
        n_samples: int,
        songs_per_sample: int,
        measures_per_song: int,
        reshaper: Callable[[np.ndarray], np.ndarray],
        track_names: List[str],
    ):
        """Empty bars ratio
        Args:
            n_samples (int): count of total samples
            songs_per_sample (int): songs per sample
            reshaper (Callable[[np.ndarray], np.ndarray]): reshape function.
                it convert pianoroll shape into (song, track, timestep, pitch)
            track_names (List[str]): list of track name
        """
        self.results = np.zeros((n_samples, len(track_names)))
        self.songs_per_sample = songs_per_sample
        self.measures_per_song = measures_per_song
        self.reshaper =reshaper
        self.track_names = track_names

    def __call__(self, idx: int, pianoroll: np.ndarray) -> None:
        """calculate empty bars ratio

        Args:
            idx (int): index of the sample
            pianoroll (np.ndarray): pianoroll of n songs.
        """
        pianoroll = self.reshaper(pianoroll)
        pianoroll = _add_measure_axis(pianoroll, self.measures_per_song)
        # (songs, track, measure, timestep, pitch)

        result = np.average(
            np.all(pianoroll == 0, axis=(3, 4)).astype(np.float32),
            axis=2,
        )
        self.results[
            idx * self.songs_per_sample:
                (idx + 1) * self.songs_per_sample] = result

    def to_dict(self):
        results = _to_dict_average_over_track(self.track_names, self.results)
        return {"empty_bars": results}


class UsedPitchClasses(EvaluateUnit):
    def __init__(
        self,
        n_samples: int,
        songs_per_sample: int,
        measures_per_song: int,
        reshaper: Callable[[np.ndarray], np.ndarray],
        track_names: List[str],
    ):
        """Used pitch classes
        Args:
            n_samples (int): count of total samples
            songs_per_sample (int): songs per sample
            reshaper (Callable[[np.ndarray], np.ndarray]): reshape function.
                it convert pianoroll shape into (song, track, timestep, pitch)
            track_names (List[str]): list of track name
        """
        self.results = np.zeros((n_samples, len(track_names)))
        self.songs_per_sample = songs_per_sample
        self.measures_per_song = measures_per_song
        self.reshaper =reshaper
        self.track_names = track_names

    def __call__(self, idx: int, pianoroll: np.ndarray) -> None:
        """calculate used pitch classes

        Args:
            idx (int): index of the sample
            pianoroll (np.ndarray): pianoroll of n songs.
        """
        pianoroll = self.reshaper(pianoroll)
        # (songs, track, timestep, pitch)

        rem = pianoroll.shape[3] % 12
        # process octave fraction
        if (rem != 0):
            reminder = pianoroll[..., -rem:]
            pianoroll = pianoroll[..., :-rem]
        pianoroll = np.reshape(
            pianoroll,
            (pianoroll.shape[0], pianoroll.shape[1], pianoroll.shape[2], 12, -1)
        )

        pianoroll_12_pitches = np.sum(pianoroll, axis=4)

        if (rem != 0):
            pianoroll_12_pitches[..., :rem] += reminder
        pianoroll_12_pitches = _add_measure_axis(
            pianoroll_12_pitches, self.measures_per_song)

        pianoroll_12_pitches = np.sum(pianoroll_12_pitches, axis=3)
        # (song, track, measure, 12)
        measure_used_pitches = np.count_nonzero(pianoroll_12_pitches, axis=3)
        # # (song, track, measure)
        result = np.average(measure_used_pitches, axis=2)
        self.results[
            idx * self.songs_per_sample:
                (idx + 1) * self.songs_per_sample] = result

    def to_dict(self):
        results = _to_dict_average_over_track(self.track_names, self.results)
        return {"used_pitch_classes": results}
# TODO: implement each methods
