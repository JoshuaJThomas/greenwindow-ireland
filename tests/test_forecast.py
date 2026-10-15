import numpy as np
import pandas as pd
import pytest

from greenwindow.forecast import HORIZON, naive_forecast, summarise
from greenwindow.scheduler import BLOCK_STEPS


def test_the_shipped_checkpoint_is_the_small_student(forecaster):
    # 1.6M parameters: the 30x reduction from the 48M Chronos-Bolt teacher
    assert 1.5e6 < forecaster.n_params < 1.7e6


def test_forecast_on_real_data_is_sane(forecaster, history):
    fc = forecaster.predict(history)
    assert fc.values.shape == (HORIZON,)
    assert np.isfinite(fc.values).all()
    assert (fc.values > 0).all() and (fc.values < 800).all()
    assert fc.origin == history.dropna().index[-1]
    assert fc.stamps[0] == fc.origin + pd.Timedelta("15min")


def test_cpu_inference_is_fast(forecaster, history):
    forecaster.predict(history)                  # warm-up
    best = min(forecaster.predict(history).latency_ms for _ in range(5))
    assert best < 500                            # ~20-60 ms on a laptop


def test_forecast_is_deterministic(forecaster, history):
    a, b = forecaster.predict(history), forecaster.predict(history)
    assert np.array_equal(a.values, b.values)


def test_too_little_history_is_refused(forecaster, history):
    with pytest.raises(ValueError, match="need 512"):
        forecaster.predict(history.dropna().iloc[-100:])


def test_window_never_starts_in_the_past(forecaster, history):
    fc = forecaster.predict(history)
    now = fc.origin + pd.Timedelta(hours=10)     # EirGrid lagging 10 h
    start, _ = fc.best_window(after=now)
    assert fc.stamps[start] - pd.Timedelta("15min") >= now


def test_summary_is_consistent(forecaster, history):
    fc = forecaster.predict(history)
    out = summarise(fc, history, forecaster.n_params)
    w = out["window"]
    start, mean_ci = fc.best_window()
    assert w["mean_g_per_kwh"] == pytest.approx(mean_ci, abs=0.05)
    assert pd.Timestamp(w["end"]) - pd.Timestamp(w["start"]) == pd.Timedelta(hours=3)
    assert w["saving_vs_now_pct"] >= 0           # waiting never looks worse
    assert len(out["forecast"]) == len(out["naive"]) == HORIZON


def test_naive_baseline_repeats_yesterday(history):
    naive = naive_forecast(history)
    last_day = history.dropna().iloc[-96:].to_numpy()
    assert np.array_equal(naive[:96], last_day) and np.array_equal(naive[96:], last_day)
    assert BLOCK_STEPS == 12
