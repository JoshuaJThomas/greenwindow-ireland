# GreenWindow Ireland

[![ci](https://github.com/JoshuaJThomas/greenwindow-ireland/actions/workflows/ci.yml/badge.svg)](https://github.com/JoshuaJThomas/greenwindow-ireland/actions/workflows/ci.yml)

**Finds the cleanest 3-hour window on the Irish electricity grid over the next 48 hours**: the best time to run a wash, charge an EV or schedule anything power-hungry.

I distilled a fine-tuned Chronos-Bolt forecaster (about 48M parameters) into a 1.6M-parameter PatchTST that runs on a CPU. On the project benchmark the student matches its teacher's 24-hour accuracy (MASE 0.57 to 0.58 for both) at about 3% of the size. It runs every morning on live EirGrid data for [my portfolio site](https://joshuajthomas.github.io/), which scores each forecast against what the grid then did. Over its first 14 days its mean error was 36.5 g CO₂/kWh, against 43.9 for "same as yesterday".

## Use it

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[serve]"

greenwindow forecast          # live forecast and the cleanest window
greenwindow serve             # HTTP API on localhost:8000
greenwindow export-onnx       # write assets/student.onnx
```

```text
$ greenwindow forecast
Last EirGrid reading: Thu 15 Oct 12:45
Cleanest 3 hours:     Sat 08:00-11:00, 148 g CO2/kWh (23% below running now)
Model: 1,620,736 parameters, 8 ms on CPU
```

Or with Docker:

```bash
docker build -t greenwindow-api .
docker run -p 8000:8000 greenwindow-api
curl localhost:8000/forecast
```

| Endpoint | What it does |
|---|---|
| `GET /health` | Model loaded, parameter count |
| `GET /forecast` | Fetches EirGrid, forecasts 48 h, returns the cleanest 3-hour window and the saving against running now. EirGrid data is cached for 15 minutes. |
| `POST /forecast` | Same, from history you send (`{"end": ..., "values": [...]}`), for offline use |

The Streamlit page is still here: `pip install -e ".[app]"`, then `streamlit run app.py`.

## How it works

```text
EirGrid chart API ──> eirgrid.py ──> forecast.py ─────────> scheduler.py ──> api.py / cli.py / app.py
 (15-min, Irish        retries,       512 steps in,          cheapest          JSON, terminal,
  local time)          UTC grid,      192 out (48 h);        unbroken          Streamlit
                       gap rules      PatchTST or ONNX       3-hour block
```

| Module | Responsibility |
|---|---|
| `eirgrid.py` | Calls the dashboard API and retries the empty 200s it sometimes sends. Converts Irish local time to a regular 15-minute UTC grid, so the clock change never shifts the model's input. Fills gaps of up to an hour and leaves longer outages missing. |
| `features.py` | Puts the target first and adds time-of-day and day-of-week encodings, standardised with training-split statistics only. Training and live inference share this code. |
| `model.py` | PatchTST (Nie et al., ICLR 2023) with RevIN, so a dirtier or more volatile grid gives a forecast shifted or stretched the same way. A test checks this. |
| `forecast.py` | Loads the checkpoint once, then forecasts in about 4 ms on one CPU thread. Also has the "same as yesterday" baseline and the window summary. |
| `scheduler.py` | Picks the cheapest contiguous 3-hour block. It has to be one block: a spiky forecast can't stitch together scattered dips. It never schedules before the current time, even when EirGrid lags. |
| `onnx_export.py` | Exports to ONNX. ONNX Runtime gives the same forecast (max difference under 0.0001 g/kWh) in 2.4 ms, and drops the ~200 MB PyTorch dependency for deployment. |
| `windows.py` | Lazy sliding windows for training. It only keeps windows whose target is fully observed, so missing readings are never invented. |

Training (teacher fine-tuning, distillation and the benchmark against Chronos-Bolt and naive baselines) was part of the MSc project and isn't in this repo. This repo is the inference side: the trained student, and everything needed to run and serve it.

## Tests

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"
pytest              # 39 tests, offline, about 3 s
pytest -m network   # also checks the live EirGrid API still answers as expected
ruff check .
```

The suite runs on a real EirGrid response recorded on 15 October 2026 (`tests/data/`), so it needs no network. It covers:

- the scheduler, including the no-stitching rule and the deferral window;
- parsing, with Irish summer time, three payload shapes and the gap rules;
- retry behaviour;
- the model's shift and scale behaviour;
- a CPU smoke test of the shipped checkpoint;
- ONNX against PyTorch parity;
- the API, including caching, a 422 for short history and a 502 when EirGrid is down.

CI runs everything on Python 3.11 to 3.13. It then builds the Docker image and checks `/health`. A weekly live run catches EirGrid API changes before the daily forecast does.

## Layout

```text
src/greenwindow/     the package (above)
src/models, src/analysis
                     thin re-exports of the old module paths, kept because
                     the portfolio site's daily job imports them
assets/              student.pt (1.6M params) and stats.json
tests/               pytest suite and the recorded EirGrid sample
app.py               Streamlit page
Dockerfile           CPU-only API image
```

## Why

Ireland's grid carbon intensity swings heavily with wind. In one recent five-day stretch a kilowatt-hour ranged from 84 to 248 g CO₂. Moving a flexible load, like an EV charge or a dishwasher, into the right 3-hour window is one of the cheapest carbon savings a household has. This tool shows where that window is.
