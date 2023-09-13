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


def _drum_pattern_mask(n_timesteps: int, tolerance=0.1):
    """Return a drum pattern mask with the given tolerance.
    code from
    https://github.com/salu133445/musegan/blob/main/src/musegan/metrics.py#L77

    Args:
        n_timesteps (int): number of timesteps in measure
        tolerance (float): weight for non on-beat timestep

    Returns:
        np.ndarray: drum pattern mask. shape=(n_timesteps,)
    """
    if n_timesteps not in (96, 48, 24, 72, 36, 64, 32, 16):
        raise ValueError("Unsupported number of timesteps for the drum in "
                         "pattern metric.")
    if n_timesteps == 96:
        drum_pattern_mask = np.tile(
            [1., tolerance, 0., 0., 0., tolerance], 16)
    elif n_timesteps == 48:
        drum_pattern_mask = np.tile([1., tolerance, tolerance], 16)
    elif n_timesteps == 24:
        drum_pattern_mask = np.tile([1., tolerance, tolerance], 8)
    elif n_timesteps == 72:
        drum_pattern_mask = np.tile(
            [1., tolerance, 0., 0., 0., tolerance], 12)
    elif n_timesteps == 36:
        drum_pattern_mask = np.tile([1., tolerance, tolerance], 12)
    elif n_timesteps == 64:
        drum_pattern_mask = np.tile([1., tolerance, 0., tolerance], 16)
    elif n_timesteps == 32:
        drum_pattern_mask = np.tile([1., tolerance], 16)
    elif n_timesteps == 16:
        drum_pattern_mask = np.tile([1., tolerance], 8)
    return drum_pattern_mask


class DrumPattern(EvaluateUnit):
    def __init__(
        self,
        n_samples: int,
        songs_per_sample: int,
        measures_per_song: int,
        reshaper: Callable[[np.ndarray], np.ndarray],
        timesteps_per_measure: int,
    ):
        """Drum pattern
        Args:
            n_samples (int): count of total samples
            songs_per_sample (int): songs per sample
            reshaper (Callable[[np.ndarray], np.ndarray]): reshape function.
                it convert pianoroll shape into (song, timestep, pitch). it has
                only drum track
            timesteps_per_measure (int): timesteps per measure
        """
        self.results = np.zeros(n_samples)
        self.songs_per_sample = songs_per_sample
        self.measures_per_song = measures_per_song
        self.reshaper =reshaper
        self.drum_pattern_mask = np.tile(
            _drum_pattern_mask(timesteps_per_measure),
            (songs_per_sample, measures_per_song, 1))

    def __call__(self, idx: int, pianoroll: np.ndarray) -> None:
        """calculate drum pattern

        Args:
            idx (int): index of the sample
            pianoroll (np.ndarray): pianoroll of n songs.
        """
        pianoroll = self.reshaper(pianoroll)
        # (song, timestep, pitch)
        pianoroll = pianoroll.reshape(
            pianoroll.shape[0], self.measures_per_song, -1, pianoroll.shape[2]
        )
        # (songs, measure, timestep, pitch)

        n_notes = np.count_nonzero(pianoroll, axis=(2, 3))
        pianoroll = np.sum(pianoroll, axis=3)
        measure_drum_pattern = np.sum(
            pianoroll * self.drum_pattern_mask, axis=2)
        # (songs, measure)
        result = np.nan_to_num(measure_drum_pattern / n_notes)

        result = np.average(result, axis=1)

        self.results[
            idx * self.songs_per_sample:
                (idx + 1) * self.songs_per_sample] = result

    def to_dict(self):
        return {"drum_pattern": np.average(self.results).item()}
# TODO: implement each methods
