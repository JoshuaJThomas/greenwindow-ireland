"""Client for the EirGrid smart grid dashboard chart API.

All-island series arrive in 15-minute steps, stamped in Irish local time
with no offset. This module turns them into a UTC-indexed series on a
regular 15-minute grid, so DST changes and short gaps never shift the
model's input window.
"""
import datetime as dt
import time
from collections.abc import Callable

import pandas as pd
import requests

API = "https://www.smartgriddashboard.com/api/chart/"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


class EmptyResponse(RuntimeError):
    """The API answered 200 with no rows (it does this transiently)."""


def extract_rows(payload) -> list[dict]:
    """The API has answered {'Rows': [...]}, {'items': [...]} and a bare
    list at different times; accept all three."""
    if isinstance(payload, dict):
        return (payload.get("items") or payload.get("Rows")
                or payload.get("rows") or [])
    return payload or []


def parse_rows(rows: list[dict]) -> pd.Series:
    """Rows -> float series on a regular 15-min UTC grid.

    Gaps of up to an hour are interpolated; longer outages stay NaN so
    they can't pass for real readings.
    """
    if not rows:
        raise EmptyResponse("no rows to parse")
    frame = pd.DataFrame(rows)
    tcol = next(c for c in frame.columns
                if c.lower() in ("effectivetime", "time", "datetime"))
    vcol = next(c for c in frame.columns
                if c.lower() in ("value", "fieldvalue"))
    ts = pd.to_datetime(frame[tcol], dayfirst=True)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("Europe/Dublin", ambiguous=True,
                               nonexistent="shift_forward")
    series = pd.Series(pd.to_numeric(frame[vcol], errors="coerce").to_numpy(),
                       index=ts.dt.tz_convert("UTC")).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series.asfreq("15min").interpolate(limit=4, limit_area="inside")


def fetch_series(chart_type: str, area: str, days_back: int,
                 days_ahead: int, *, today: dt.date | None = None,
                 retries: int = 4, get: Callable = requests.get,
                 sleep: Callable[[float], None] = time.sleep) -> pd.Series:
    """Fetch one series. Retries with backoff, because the API sometimes
    returns an empty 200 first and the full data on the next try."""
    today = today or dt.date.today()
    params = {"region": "ALL", "chartType": chart_type, "dateRange": "week",
              "dateFrom": (today - dt.timedelta(days=days_back)).isoformat(),
              "dateTo": (today + dt.timedelta(days=days_ahead)).isoformat(),
              "areas": area}
    last_err: Exception = EmptyResponse(f"{area}: no attempts made")
    for attempt in range(retries):
        try:
            r = get(API, params=params, timeout=40,
                    headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            rows = extract_rows(r.json())
            if rows:
                return parse_rows(rows)
            last_err = EmptyResponse(f"{area}: empty response")
        except (requests.RequestException, ValueError) as exc:
            last_err = exc
        if attempt < retries - 1:
            sleep(5 * (attempt + 1))
    raise last_err


def fetch_ci_history(days: int = 6, **kwargs) -> pd.Series:
    """Recent all-island carbon intensity (g CO2/kWh).

    dateRange=week spans 7 days from dateFrom, so days=7 stops at last
    midnight and the forecast would start from stale data. days=6 reaches
    the latest reading and still covers the 512-step (~5.3-day) lookback.
    """
    return fetch_series("co2", "co2intensity", days, 1, **kwargs)
