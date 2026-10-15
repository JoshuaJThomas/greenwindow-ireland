"""Inference: run the distilled PatchTST student on recent history.

Forecaster loads the checkpoint once and is then cheap to call, which is
what the API and the daily job need. Everything runs on CPU on purpose:
the point of distilling the 48M-parameter Chronos-Bolt teacher was a
model that runs anywhere, including a free CI runner.
"""
import json
import os
import pathlib
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .features import TARGET, add_time_features
from .model import PatchTST
from .scheduler import BLOCK_STEPS, choose_block

LOOKBACK = 512          # 5.3 days of 15-min steps in
HORIZON = 192           # 48 hours out
STEP = pd.Timedelta("15min")


def default_assets_dir() -> pathlib.Path:
    """$GREENWINDOW_ASSETS, else assets/ at the repo root (the checkpoint
    is 6 MB, so it sits beside the package rather than inside it)."""
    env = os.environ.get("GREENWINDOW_ASSETS")
    if env:
        return pathlib.Path(env)
    return pathlib.Path(__file__).resolve().parents[2] / "assets"


@dataclass
class Forecast:
    origin: pd.Timestamp    # last observed step (UTC)
    values: np.ndarray      # g CO2/kWh for origin+1 .. origin+len(values)
    latency_ms: float

    @property
    def stamps(self) -> pd.DatetimeIndex:
        """End of each forecast step, matching EirGrid's timestamps."""
        return pd.date_range(self.origin + STEP, periods=len(self.values),
                             freq=STEP)

    def first_step(self, after: pd.Timestamp | None = None) -> int:
        """Index of the first step starting no earlier than `after`.
        EirGrid often lags by an hour or two, so callers pass wall-clock
        now; None means straight after the last reading."""
        if after is None:
            return 0
        return max(0, int(np.ceil((after - self.origin) / STEP)))

    def best_window(self, after: pd.Timestamp | None = None,
                    block: int = BLOCK_STEPS) -> tuple[int, float]:
        """Index and mean intensity of the cleanest contiguous block
        starting no earlier than `after`."""
        skip = self.first_step(after)
        live = self.values[skip:]
        if len(live) < block:
            raise ValueError("forecast is too old to schedule from")
        start = skip + choose_block(live, window=len(live), block=block)
        return start, float(self.values[start:start + block].mean())


class Forecaster:
    def __init__(self, model: PatchTST, mean: pd.Series, std: pd.Series,
                 lookback: int = LOOKBACK, horizon: int = HORIZON):
        self.model = model.eval()
        self.mean, self.std = mean, std
        self.columns = list(mean.index)
        self.lookback, self.horizon = lookback, horizon

    @classmethod
    def from_assets(cls, assets: str | pathlib.Path | None = None
                    ) -> "Forecaster":
        assets = pathlib.Path(assets) if assets else default_assets_dir()
        stats = json.loads((assets / "stats.json").read_text())
        ckpt = torch.load(assets / "student.pt", map_location="cpu",
                          weights_only=False)
        cfg = ckpt["config"]
        model = PatchTST(cfg["lookback"], cfg["horizon"], len(stats["mean"]),
                         d_model=cfg["d_model"], n_heads=4,
                         n_layers=cfg["n_layers"], d_ff=2 * cfg["d_model"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, pd.Series(stats["mean"]), pd.Series(stats["std"]),
                   lookback=cfg["lookback"],
                   horizon=min(cfg["horizon"], HORIZON))

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    def prepare(self, history: pd.Series) -> np.ndarray:
        """History -> standardised (1, lookback, channels) float32 input,
        built with the same features and statistics as training."""
        tail = history.dropna().iloc[-self.lookback:]
        if len(tail) < self.lookback:
            raise ValueError(
                f"need {self.lookback} observed steps, got {len(tail)}")
        feats = add_time_features(tail.to_frame(TARGET))[self.columns]
        z = (feats - self.mean) / self.std
        return z.to_numpy(dtype=np.float32)[None]

    def to_grams(self, z: np.ndarray) -> np.ndarray:
        """Standardised target channel -> g CO2/kWh."""
        return (z * self.std[TARGET] + self.mean[TARGET])[:self.horizon]

    def predict(self, history: pd.Series) -> Forecast:
        x = torch.from_numpy(self.prepare(history))
        with torch.no_grad():
            t0 = time.perf_counter()
            out = self.model(x)
            latency = (time.perf_counter() - t0) * 1000.0
        return Forecast(origin=history.dropna().index[-1],
                        values=self.to_grams(out[0, :, 0].numpy()),
                        latency_ms=latency)


def naive_forecast(history: pd.Series, horizon: int = HORIZON) -> np.ndarray:
    """'Same as yesterday': the last 24 h repeated. The baseline the
    student has to beat."""
    last_day = history.dropna().iloc[-96:].to_numpy()
    return np.tile(last_day, int(np.ceil(horizon / 96)))[:horizon]


def summarise(fc: Forecast, history: pd.Series, params: int,
              now: pd.Timestamp | None = None) -> dict:
    """The forecast plus the cleanest window and its saving against
    running straight away, as plain JSON-ready data."""
    start, mean_ci = fc.best_window(after=now)
    first = fc.first_step(now)
    now_ci = float(fc.values[first:first + BLOCK_STEPS].mean())
    stamps = fc.stamps          # step ends; a block starts one step earlier
    return {
        "origin": fc.origin.isoformat(),
        "step_minutes": 15,
        "forecast": [round(float(v), 1) for v in fc.values],
        "naive": [round(float(v), 1) for v in naive_forecast(history)],
        "window": {
            "start": (stamps[start] - STEP).isoformat(),
            "end": stamps[start + BLOCK_STEPS - 1].isoformat(),
            "mean_g_per_kwh": round(mean_ci, 1),
            "saving_vs_now_pct": (round(100 * (now_ci - mean_ci) / now_ci, 1)
                                  if now_ci > 0 else 0.0),
        },
        "latency_ms": round(fc.latency_ms, 2),
        "model_params": params,
    }
