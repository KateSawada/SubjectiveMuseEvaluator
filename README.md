# Reshaper
input shape: any(depends on each reshaper)
output shape: (n_songs, n_measures, timesteps, n_pitch, n_tracks), dtype=bool

musegan: (n_songs, n_measures, timesteps, n_pitch, n_tracks)

# Metrics
input shape: (n_songs, n_measures, timesteps, n_pitch, n_tracks), dtype=bool


# example

```python
import glob
import numpy as np
import SubjectiveMuseEvaluator as SME

# load pianoroll from npy in given directory
filenames = glob.glob("outputs/debug/eval_debug/npys/*.npy")

# configuration
timestep_per_measure = 48
beat_per_measure = 4
track_names = ["drm", "pf", "gt", "ba", "str"]

# build Evaluator
postprocess = SME.stats.CalculateTrackStatistics()
loader = SME.loader.NDarrayFromNpy()
reshaper = SME.reshaper.LpdCleansedReshaper(
    input_shape=(4, 48, 84, 5),
    songs_per_sample=1,
    n_tracks=5,
    measures_per_song=4,
    timesteps_per_measure=48,
    n_pitches=84,
    hard_threshold=0.5,
)
methods = [
    SME.metrics.UsedPitchClasses(
        n_samples=len(filenames),
        songs_per_sample=1,
        measures_per_song=4,
        reshaper=reshaper,
        track_names=track_names,
        postprocess=postprocess,
    ),
    SME.metrics.EmptyBarRate(
        n_samples=len(filenames),
        songs_per_sample=1,
        measures_per_song=4,
        reshaper=reshaper,
        track_names=track_names,
        postprocess=postprocess,
    ),
    SME.metrics.DrumPattern(
        n_samples=len(filenames),
        songs_per_sample=1,
        measures_per_song=4,
        reshaper=reshaper,
        track_names=track_names,
        postprocess=postprocess,
        target_track_index=0,
        timestep_per_measure=timestep_per_measure
    ),
    SME.metrics.TonalDistance(
        n_samples=len(filenames),
        songs_per_sample=1,
        measures_per_song=4,
        reshaper=reshaper,
        track_names=track_names,
        postprocess=postprocess,
        timestep_per_measure=timestep_per_measure,
        beat_per_measure=beat_per_measure,
    )
]

evaluator = SME.Evaluator(
    methods=methods,
    filenames=filenames,
    loader=loader,
)
evaluator.run("outputs/debug/eval_debug/result.yaml")
```
