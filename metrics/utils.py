from typing import (
    List,
    Dict,
)

import numpy as np


def to_dict_average_over_track(
    track_names: List[str],
    results: np.ndarray,
) -> Dict[str, float]:
    results_dict = {}
    for i in range(len(track_names)):
        results_dict[track_names[i]] = np.average(results[:, i]).item()
    return results_dict


def to_chroma(pianoroll: np.ndarray) -> np.ndarray:
    """convert pianoroll into chroma

    Args:
        pianoroll (np.ndarray): pianoroll.
            shape=(n_songs, n_measures, timesteps, n_pitch, n_tracks)

    Returns:
        np.ndarray: chroma feature tensor.
            shape=(n_songs, n_measures, timesteps, 12, n_tracks)
    """
    remainder = pianoroll.shape[3] % 12
    if (remainder):
        pianoroll = np.pad(
            pianoroll,
            ((0, 0), (0, 0), (0, 0), (0, 12 - remainder), (0, 0))
        )
    reshaped = pianoroll.reshape(
        pianoroll.shape[0],
        pianoroll.shape[1],
        pianoroll.shape[2],
        pianoroll.shape[3] // 12,
        12,
        pianoroll.shape[4],
    ).transpose(0, 1, 2, 4, 3, 5)
    return np.sum(reshaped, axis=4)


def drum_pattern_mask(n_timesteps, tolerance=0.1):
    """Return a drum pattern mask with the given tolerance."""
    if n_timesteps not in (96, 48, 24, 72, 36, 64, 32, 16):
        raise ValueError(
            "Unsupported number of timesteps for the drum in "
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


def create_tonal_matrix(r1=1.0, r2=1.0, r3=0.5):
    """Compute and return a tonal matrix for computing the tonal distance
    [1]. Default argument values are set as suggested by the paper.

    [1] Christopher Harte, Mark Sandler, and Martin Gasser. Detecting
    harmonic change in musical audio. In Proc. ACM MM Workshop on Audio and
    Music Computing Multimedia, 2006.
    """
    tonal_matrix = np.empty((6, 12))
    tonal_matrix[0] = r1 * np.sin(np.arange(12) * (7. / 6.) * np.pi)
    tonal_matrix[1] = r1 * np.cos(np.arange(12) * (7. / 6.) * np.pi)
    tonal_matrix[2] = r2 * np.sin(np.arange(12) * (3. / 2.) * np.pi)
    tonal_matrix[3] = r2 * np.cos(np.arange(12) * (3. / 2.) * np.pi)
    tonal_matrix[4] = r3 * np.sin(np.arange(12) * (2. / 3.) * np.pi)
    tonal_matrix[5] = r3 * np.cos(np.arange(12) * (2. / 3.) * np.pi)
    return tonal_matrix
