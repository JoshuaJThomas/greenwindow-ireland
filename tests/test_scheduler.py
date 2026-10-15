import numpy as np
import pytest

from greenwindow.scheduler import BLOCK_STEPS, block_cost, choose_block, savings_pct


def test_picks_the_lowest_block():
    f = np.full(96, 300.0)
    f[40:52] = 100.0
    assert choose_block(f, window=96) == 40


def test_needs_one_unbroken_block_not_scattered_dips():
    # Twelve isolated 0 g/kWh dips would win if blocks could be stitched
    # together; a steady 150 g/kWh stretch is the real best 3 hours.
    f = np.full(192, 300.0)
    f[0:96:8] = 0.0
    f[120:132] = 150.0
    start = choose_block(f, window=192)
    assert start == 120
    assert f[start:start + BLOCK_STEPS].mean() == 150.0


def test_respects_the_deferral_window():
    f = np.full(192, 300.0)
    f[100:112] = 50.0          # best overall, but outside an 8 h window
    f[10:22] = 200.0
    assert choose_block(f, window=32) == 10


def test_window_longer_than_forecast_is_an_error():
    with pytest.raises(ValueError):
        choose_block(np.zeros(10), window=20)


def test_block_cost_refuses_to_invent_missing_readings():
    signal = np.full(24, 100.0)
    signal[5] = np.nan
    assert np.isnan(block_cost(signal, 0))
    assert block_cost(signal, 6) == 100.0


def test_savings_never_beat_the_oracle():
    rng = np.random.default_rng(0)
    actual = 250 + 80 * np.sin(np.linspace(0, 4 * np.pi, 192)) + rng.normal(0, 10, 192)
    forecast = actual + rng.normal(0, 25, 192)      # an imperfect forecast
    out = savings_pct(forecast, actual, window=192)
    assert out["oracle_pct"] >= out["saved_pct"]
    perfect = savings_pct(actual, actual, window=192)
    assert perfect["saved_pct"] == pytest.approx(perfect["oracle_pct"])
