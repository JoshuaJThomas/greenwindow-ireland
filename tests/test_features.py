import numpy as np
import pandas as pd

from greenwindow.features import TARGET, add_time_features, feature_frame, fit_stats
from greenwindow.windows import WindowSet


def frame(n: int, start="2026-01-05 00:00", values=None) -> pd.DataFrame:
    index = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    v = np.arange(n, dtype=float) if values is None else values
    return pd.DataFrame({"wind": v * 2, TARGET: v}, index=index)


def test_calendar_features_are_on_the_unit_circle():
    f = add_time_features(frame(96 * 7))
    assert np.allclose(f.tod_sin**2 + f.tod_cos**2, 1)
    assert np.allclose(f.dow_sin**2 + f.dow_cos**2, 1)


def test_time_of_day_repeats_daily_and_day_of_week_does_not():
    f = add_time_features(frame(96 * 2))
    assert np.allclose(f.tod_sin.iloc[:96].to_numpy(), f.tod_sin.iloc[96:].to_numpy())
    assert not np.allclose(f.dow_sin.iloc[:96].to_numpy(), f.dow_sin.iloc[96:].to_numpy())


def test_target_is_always_channel_zero():
    assert feature_frame(frame(10)).columns[0] == TARGET


def test_statistics_come_from_the_training_split_only():
    train = frame(200)
    test = frame(200, start="2026-02-01", values=np.full(200, 1e6))
    stats = fit_stats(train)
    ws = WindowSet(test, stats, lookback=16, horizon=4)
    # test-split values are huge in training units, not re-centred to ~0
    assert ws.matrix[:, 0].min() > 100


def test_windows_skip_any_gap_in_the_target():
    values = np.arange(100, dtype=float)
    values[50] = np.nan
    ws = WindowSet(frame(100, values=values), fit_stats(frame(100)), lookback=16, horizon=4)
    for o in ws.origins:
        covered = range(o - 15, o + 5)
        assert 50 not in covered
    x, y = ws.get(0)
    assert x.shape == (16, len(ws.columns)) and y.shape == (4,)
