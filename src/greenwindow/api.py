"""HTTP API: the forecast and the cleanest window as JSON.

    uvicorn greenwindow.api:app            # or: greenwindow serve

GET  /health     model loaded, parameter count
GET  /forecast   live: fetch EirGrid, forecast, pick the window
POST /forecast   offline: forecast from history you send

The model loads once at startup. Live EirGrid data is cached for 15
minutes, the API's own refresh rate, so a busy endpoint doesn't hammer
EirGrid.
"""
import time
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .eirgrid import fetch_ci_history
from .forecast import Forecaster, summarise

CACHE_SECONDS = 900


class HistoryIn(BaseModel):
    """Carbon intensity readings, oldest first, 15 minutes apart."""
    end: datetime = Field(description="time of the last reading (UTC if no offset)")
    values: list[float | None] = Field(min_length=1)


class Window(BaseModel):
    start: str
    end: str
    mean_g_per_kwh: float
    saving_vs_now_pct: float


class ForecastOut(BaseModel):
    origin: str
    step_minutes: int
    forecast: list[float]
    naive: list[float]
    window: Window
    latency_ms: float
    model_params: int


def history_from(body: HistoryIn) -> pd.Series:
    end = pd.Timestamp(body.end)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    index = pd.date_range(end=end, periods=len(body.values), freq="15min")
    return pd.Series(body.values, index=index, dtype=float)


def create_app(forecaster: Forecaster | None = None, fetch=fetch_ci_history,
               clock=lambda: pd.Timestamp.now(tz="UTC")) -> FastAPI:
    """`fetch` and `clock` are injectable so tests run offline."""
    state: dict = {"forecaster": forecaster, "cached": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["forecaster"] is None:
            state["forecaster"] = Forecaster.from_assets()
        yield

    app = FastAPI(title="GreenWindow Ireland", version=__version__,
                  lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_params": state["forecaster"].n_params}

    @app.get("/forecast", response_model=ForecastOut)
    def live() -> dict:
        cached = state["cached"]
        if cached is None or time.monotonic() - cached[0] > CACHE_SECONDS:
            try:
                state["cached"] = cached = (time.monotonic(), fetch())
            except Exception as exc:
                raise HTTPException(502, f"EirGrid unavailable: {exc}") from exc
        history = cached[1]
        model = state["forecaster"]
        try:
            fc = model.predict(history)
            return summarise(fc, history, model.n_params,
                             now=clock())
        except ValueError as exc:     # feed too short or too far behind
            raise HTTPException(503, str(exc)) from exc

    @app.post("/forecast", response_model=ForecastOut)
    def offline(body: HistoryIn) -> dict:
        history = history_from(body)
        model = state["forecaster"]
        try:
            return summarise(model.predict(history), history, model.n_params)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()
