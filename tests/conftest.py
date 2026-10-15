import json
import pathlib

import pytest

from greenwindow.eirgrid import extract_rows, parse_rows
from greenwindow.forecast import Forecaster

DATA = pathlib.Path(__file__).parent / "data"
SAMPLE = DATA / "eirgrid_co2_2026-10-15.json"   # a real API response


@pytest.fixture(scope="session")
def raw_payload() -> dict:
    return json.loads(SAMPLE.read_text())


@pytest.fixture(scope="session")
def history(raw_payload):
    """Six and a half days of real all-island carbon intensity."""
    return parse_rows(extract_rows(raw_payload))


@pytest.fixture(scope="session")
def forecaster() -> Forecaster:
    return Forecaster.from_assets()
