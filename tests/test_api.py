import pandas as pd
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from greenwindow.api import create_app  # noqa: E402

# The recorded sample ends at 12:45 UTC on 15 Oct; serve it an hour later,
# as if EirGrid were lagging.
NOW = pd.Timestamp("2026-10-15 13:45", tz="UTC")


@pytest.fixture()
def client(forecaster, history):
    calls = []

    def fetch():
        calls.append(1)
        return history

    app = create_app(forecaster=forecaster, fetch=fetch, clock=lambda: NOW)
    with TestClient(app) as c:
        c.calls = calls
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_params"] > 1_000_000


def test_offline_forecast_from_posted_history(client, history):
    h = history.dropna().iloc[-600:]
    body = {"end": h.index[-1].isoformat(), "values": h.round(1).tolist()}
    r = client.post("/forecast", json=body)
    assert r.status_code == 200
    out = r.json()
    assert len(out["forecast"]) == 192
    assert pd.Timestamp(out["origin"]) == h.index[-1]


def test_short_history_gets_a_clear_422(client):
    r = client.post("/forecast", json={"end": "2026-10-15T12:00:00Z", "values": [200.0] * 50})
    assert r.status_code == 422 and "need 512" in r.json()["detail"]


def test_live_endpoint_caches_eirgrid(client):
    first = client.get("/forecast")
    second = client.get("/forecast")
    assert first.status_code == second.status_code == 200
    assert len(client.calls) == 1               # second call served from cache
    assert pd.Timestamp(first.json()["window"]["start"]) >= NOW    # never in the past


def test_eirgrid_outage_is_a_502(forecaster):
    def down():
        raise ConnectionError("timeout")

    with TestClient(create_app(forecaster=forecaster, fetch=down)) as c:
        r = c.get("/forecast")
    assert r.status_code == 502 and "EirGrid unavailable" in r.json()["detail"]
