import datetime as dt

import numpy as np
import pandas as pd
import pytest
import requests

from greenwindow.eirgrid import (
    EmptyResponse,
    extract_rows,
    fetch_ci_history,
    fetch_series,
    parse_rows,
)


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(str(self.status))

    def json(self):
        return self.payload


def test_recorded_response_parses_to_a_regular_utc_grid(history):
    assert str(history.index.tz) == "UTC"
    assert (history.index.to_series().diff().dropna() == pd.Timedelta("15min")).all()
    assert len(history.dropna()) >= 512          # enough for the lookback
    assert history.between(0, 1000).all()        # plausible g CO2/kWh


def test_irish_summer_time_is_converted_to_utc():
    rows = [{"EffectiveTime": "15-Jul-2026 12:00:00", "Value": 200}]
    assert parse_rows(rows).index[0] == pd.Timestamp("2026-07-15 11:00", tz="UTC")


@pytest.mark.parametrize("shape", ["Rows", "items", "list"])
def test_all_three_payload_shapes(raw_payload, shape):
    rows = raw_payload["Rows"]
    payload = rows if shape == "list" else {shape: rows}
    assert extract_rows(payload) == rows


def test_short_gaps_are_filled_and_long_outages_stay_missing():
    t = pd.date_range("2026-10-01", periods=40, freq="15min")
    rows = [{"EffectiveTime": ts.strftime("%d-%b-%Y %H:%M:%S"), "Value": 100}
            for i, ts in enumerate(t) if not (5 <= i < 7 or 15 <= i < 30)]
    s = parse_rows(rows)
    assert s.iloc[5:7].notna().all()             # 30 min gap: interpolated
    assert s.iloc[15:30].isna().sum() > 0        # 3.75 h outage: left as NaN


def test_retries_through_an_empty_200(raw_payload):
    replies = iter([FakeResponse({"Rows": []}), FakeResponse(raw_payload)])
    waits = []
    s = fetch_series("co2", "co2intensity", 6, 1, today=dt.date(2026, 10, 15),
                     get=lambda *a, **k: next(replies), sleep=waits.append)
    assert len(s) > 500 and waits == [5]


def test_gives_up_after_the_last_retry():
    with pytest.raises(EmptyResponse):
        fetch_series("co2", "co2intensity", 6, 1, retries=3,
                     get=lambda *a, **k: FakeResponse({"Rows": []}),
                     sleep=lambda s: None)


def test_requests_the_right_date_range(raw_payload):
    seen = {}

    def get(url, params, **kwargs):
        seen.update(params)
        return FakeResponse(raw_payload)

    fetch_ci_history(today=dt.date(2026, 10, 15), get=get)
    assert (seen["dateFrom"], seen["dateTo"]) == ("2026-10-09", "2026-10-16")


@pytest.mark.network
def test_live_api_still_answers_in_the_expected_shape():
    s = fetch_ci_history()
    assert len(s.dropna()) >= 512 and np.isfinite(s.dropna()).all()
