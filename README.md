# GreenWindow Ireland 🍃

**Finds the cleanest 3-hour window on the Irish electricity grid over the next 48 hours** — the best time to run a wash, charge an EV, or schedule anything power-hungry.

Live EirGrid generation data feeds a distilled **PatchTST** forecasting model (1.6M parameters, runs on CPU) that predicts grid carbon intensity, and a scheduler picks the greenest window. Built as part of an MSc AI project at the National College of Ireland.

## How it works

1. **Data** - live generation and demand data pulled from EirGrid's public feeds
2. **Forecast** - a distilled PatchTST transformer (`assets/student.pt`) predicts carbon intensity for the next 48 hours on CPU, no GPU needed
3. **Schedule** - `src/analysis/scheduler.py` scans the forecast for the cleanest contiguous 3-hour window
4. **UI** - a Streamlit app presents the result

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Stack

Python · PyTorch (PatchTST) · Streamlit · EirGrid public data

## Why

Ireland's grid carbon intensity swings heavily with wind. Shifting flexible loads into the right 3-hour window is one of the cheapest carbon savings available to any household — this tool makes the window visible.
