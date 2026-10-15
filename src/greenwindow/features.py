"""Model inputs: target-first column order, calendar encodings and the
train-split standardisation statistics.

Shared by training (windows.py) and inference (forecast.py), so the
features a live forecast sees are built exactly as they were in training.
"""
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
