"""Sliding-window dataset preparation for the Tier 2 deep models.

Splits are continuous 15-min ranges (gaps are NaN values, never missing
rows), so windows are plain index-aligned slices over a standardised
feature matrix. Windows are materialised lazily: the full window tensor
for the train split would be ~9 GB, while the backing matrix is ~25 MB.

Conventions fixed here (and relied on by train.py and the tests):
- channel 0 is always the target (co2intensity);
- standardisation statistics come from the TRAIN split only;
- a window is valid iff the target is finite over its full lookback and
  horizon range (covariate gaps are forward-filled then zero-filled,
  i.e. set to the training mean, so genuine outages cannot fabricate
  target values - they only blunt covariates).
"""
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET = "co2intensity"
STEPS_PER_DAY = 96


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append sin/cos encodings of time-of-day and day-of-week (UTC index)."""
    out = frame.copy()
    seconds = (frame.index.hour * 3600 + frame.index.minute * 60).to_numpy()
    day_frac = seconds / 86400.0
    week_frac = (frame.index.dayofweek.to_numpy() + day_frac) / 7.0
    out["tod_sin"] = np.sin(2 * np.pi * day_frac)
    out["tod_cos"] = np.cos(2 * np.pi * day_frac)
    out["dow_sin"] = np.sin(2 * np.pi * week_frac)
    out["dow_cos"] = np.cos(2 * np.pi * week_frac)
    return out


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Target-first column order plus time features."""
    cols = [TARGET] + [c for c in frame.columns if c != TARGET]
    return add_time_features(frame[cols])


@dataclass
class Stats:
    mean: pd.Series
    std: pd.Series


def fit_stats(train_frame: pd.DataFrame) -> Stats:
    feats = feature_frame(train_frame)
    std = feats.std().clip(lower=1e-8)
    return Stats(mean=feats.mean(), std=std)


class WindowSet:
    """Lazy (lookback, horizon) windows over one split.

    get(i) returns (x, y): x is (lookback, n_channels) float32 with all
    channels standardised; y is (horizon,) float32, the standardised
    target. Origins are the index of the last observed step; y covers
    origin+1 .. origin+horizon.
    """

    def __init__(self, frame: pd.DataFrame, stats: Stats,
                 lookback: int, horizon: int):
        self.lookback = lookback
        self.horizon = horizon
        feats = feature_frame(frame)
        self.columns = list(feats.columns)
        self.index = feats.index
        z = (feats - stats.mean) / stats.std
        target_finite = np.isfinite(z[TARGET].to_numpy())
        z = z.ffill().fillna(0.0)
        self.matrix = z.to_numpy(dtype=np.float32)
        self.target_mean = float(stats.mean[TARGET])
        self.target_std = float(stats.std[TARGET])
        # valid origin o: target finite on [o-lookback+1, o+horizon]
        run = np.convolve(target_finite.astype(np.int32),
                          np.ones(lookback + horizon, dtype=np.int32),
                          mode="valid")
        starts = np.nonzero(run == lookback + horizon)[0]
        self.origins = starts + lookback - 1

    def __len__(self) -> int:
        return len(self.origins)

    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        o = self.origins[i]
        x = self.matrix[o - self.lookback + 1:o + 1]
        y = self.matrix[o + 1:o + 1 + self.horizon, 0]
        return x, y

    def inverse_target(self, z: np.ndarray) -> np.ndarray:
        return z * self.target_std + self.target_mean


def load_windows(data_dir: str | pathlib.Path, lookback: int, horizon: int,
                 splits: tuple[str, ...] = ("train", "val", "test"),
                 univariate: bool = False,
                 ) -> dict[str, WindowSet]:
    """univariate=True drops every measured covariate, keeping the target
    plus the calendar encodings (which any univariate model can derive
    from timestamps) — the control that separates pretraining/architecture
    from covariate access."""
    data_dir = pathlib.Path(data_dir)
    frames = {s: pd.read_parquet(data_dir / f"dataset_{s}.parquet")
              for s in splits}
    if univariate:
        frames = {s: f[[TARGET]] for s, f in frames.items()}
    stats = fit_stats(frames["train"]) if "train" in frames else None
    if stats is None:
        raise ValueError("train split is required to fit statistics")
    return {s: WindowSet(f, stats, lookback, horizon)
            for s, f in frames.items()}
