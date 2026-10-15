"""GreenWindow Ireland: when is the cleanest 3-hour window in the next 48h?

Streamlit app for the club-week demo. Pulls the last ~6 days of carbon
intensity from EirGrid's dashboard chart API (the reliable endpoint the
forecast logger uses), runs the distilled student model on CPU, applies
the RQ2 scheduler and shows the recommended window with an estimated
saving for a user-chosen load (EV, home battery, appliance cycle).

Runs in two modes:
- student mode: app/assets/student.pt + stats.json exist (produced by
  python -m src.analysis.make_app_assets after RQ3 training);
- fallback mode: no checkpoint yet, forecast is seasonal-naive from the
  fetched history, clearly labelled DEMO so nobody mistakes it for the
  benchmark model.

Local run:    streamlit run app/greenwindow.py
HF Spaces:    upload the deploy/greenwindow-space/ bundle (built by
              python app/build_space.py; steps in the bundle's
              DEPLOY-STEPS.md). See BATCH-PLAN.md batch 5.
"""
import datetime as dt
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.analysis.scheduler import BLOCK_STEPS, choose_block  # noqa: E402

ASSETS = pathlib.Path(__file__).parent / "assets"
API = "https://www.smartgriddashboard.com/api/chart/"
LOOKBACK = 512          # student input steps
HORIZON = 192           # 48 h at 15 min

# Chart palette: CVD-validated (dataviz six-checks, light surface).
CI_COLOUR = "#0072B2"
WIND_COLOUR = "#B25F00"
WINDOW_COLOUR = "#009E73"

LOADS = {                # label -> kWh drawn during the 3 h block
    "EV full charge (60 kWh)": 60.0,
    "EV top-up (20 kWh)": 20.0,
    "Plug-in hybrid (12 kWh)": 12.0,
    "Home battery (13.5 kWh)": 13.5,
    "Tumble dryer cycle (2.5 kWh)": 2.5,
    "Dishwasher cycle (1.2 kWh)": 1.2,
    "Custom": None,
}

CSS = """
<style>
.gw-hero {
  background: linear-gradient(135deg, #0b3d2e 0%, #14805f 100%);
  color: #ffffff; border-radius: 16px;
  padding: 1.3rem 1.5rem 1.2rem 1.5rem; margin: 0.4rem 0 1.0rem 0;
}
.gw-hero .gw-kicker {
  font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase;
  opacity: 0.85; margin-bottom: 0.15rem;
}
.gw-hero .gw-window {
  font-size: 1.9rem; font-weight: 700; line-height: 1.15; margin: 0;
}
.gw-hero .gw-sub { font-size: 0.95rem; opacity: 0.92; margin-top: 0.4rem; }
.gw-chip {
  display: inline-block; background: rgba(255,255,255,0.16);
  border-radius: 999px; padding: 0.15rem 0.7rem; font-size: 0.82rem;
  margin-top: 0.55rem;
}
div[data-testid="stMetric"] {
  background: #f0f4f1; border-radius: 12px; padding: 0.7rem 0.9rem;
}
</style>
"""


@st.cache_data(ttl=900)
def fetch_series(chart_type: str, area: str, days_back: int,
                 days_ahead: int) -> pd.Series:
    """One series from the dashboard chart API (15-min, UTC index)."""
    today = dt.date.today()
    r = requests.get(API, params={
        "region": "ALL", "chartType": chart_type, "dateRange": "week",
        "dateFrom": (today - dt.timedelta(days=days_back)).isoformat(),
        "dateTo": (today + dt.timedelta(days=days_ahead)).isoformat(),
        "areas": area}, timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (GreenWindow demo)"})
    r.raise_for_status()
    j = r.json()   # the API answers {'Rows': [...]}, {'items': [...]}
    rows = ((j.get("items") or j.get("Rows") or j.get("rows"))
            if isinstance(j, dict) else j)         # ... or a bare list
    frame = pd.DataFrame(rows)
    time_col = next(c for c in frame.columns
                    if c.lower() in ("effectivetime", "time", "datetime"))
    val_col = next(c for c in frame.columns
                   if c.lower() in ("value", "fieldvalue"))
    ts = pd.to_datetime(frame[time_col], dayfirst=True)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("Europe/Dublin", ambiguous=True,
                               nonexistent="shift_forward")
    series = (pd.Series(frame[val_col].astype(float).to_numpy(),
                        index=ts.dt.tz_convert("UTC"))
              .sort_index())
    series = series[~series.index.duplicated(keep="last")]
    return series.asfreq("15min").interpolate(limit=4)


def fetch_ci_history(days: int = 6) -> pd.Series:
    # dateRange=week returns 7 days starting at dateFrom, so days=7 ends at
    # last midnight and the forecast starts from stale data. days=6 reaches
    # the latest reading and still covers the 512-step (~5.3-day) lookback.
    # Do not go above 6.
    return fetch_series("co2", "co2intensity", days, 1)


def student_forecast(history: pd.Series) -> tuple[np.ndarray, dict] | None:
    """Run the distilled student if its assets exist; else None.

    Returns (forecast, meta) where meta has the parameter count and the
    measured forward-pass latency for the model card.
    """
    ckpt_path, stats_path = ASSETS / "student.pt", ASSETS / "stats.json"
    if not (ckpt_path.exists() and stats_path.exists()):
        return None
    import torch
    from src.models.patchtst import PatchTST
    from src.models.windows import add_time_features

    stats = json.loads(stats_path.read_text())
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    tail = history.dropna().iloc[-LOOKBACK:]
    if len(tail) < LOOKBACK:
        return None
    feats = add_time_features(tail.to_frame("co2intensity"))
    cols = list(stats["mean"].keys())
    z = (feats[cols] - pd.Series(stats["mean"])) / pd.Series(stats["std"])
    model = PatchTST(cfg["lookback"], cfg["horizon"], len(cols),
                     d_model=cfg["d_model"], n_heads=4,
                     n_layers=cfg["n_layers"], d_ff=2 * cfg["d_model"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(z.to_numpy(dtype=np.float32))[None]
        t0 = time.perf_counter()
        out = model(x)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        pred_std = (out[:, :, 0] if out.dim() == 3 else out)[0].numpy()
    mean, std = stats["mean"]["co2intensity"], stats["std"]["co2intensity"]
    meta = {"params": sum(p.numel() for p in model.parameters()),
            "latency_ms": latency_ms}
    return (pred_std * std + mean)[:HORIZON], meta


def naive_forecast(history: pd.Series) -> np.ndarray:
    last_day = history.dropna().iloc[-96:].to_numpy()
    return np.tile(last_day, 2)[:HORIZON]


def fmt_saving(grams: float) -> str:
    grams = max(grams, 0.0)
    return f"{grams / 1000:.1f} kg CO2" if grams >= 1000 \
        else f"{grams:.0f} g CO2"


def build_chart(history: pd.Series, forecast: np.ndarray,
                stamps: pd.DatetimeIndex, block: pd.DatetimeIndex,
                origin: pd.Timestamp, wind: pd.Series | None,
                forecast_name: str):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = 2 if wind is not None else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08,
                        row_heights=[0.72, 0.28] if rows == 2 else [1.0])

    hist = history.dropna().iloc[-96:]
    local = hist.index.tz_convert("Europe/Dublin")
    fig.add_trace(go.Scatter(
        x=local, y=hist.to_numpy(), name="Actual (EirGrid)",
        line=dict(color=CI_COLOUR, width=2),
        hovertemplate="%{y:.0f} g/kWh<extra>Actual</extra>"), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=stamps.tz_convert("Europe/Dublin"), y=forecast,
        name=forecast_name,
        line=dict(color=CI_COLOUR, width=2, dash="dash"),
        hovertemplate="%{y:.0f} g/kWh<extra>Forecast</extra>"), row=1, col=1)

    if wind is not None:
        w = wind.dropna()
        w = w[(w.index >= hist.index[0]) & (w.index <= stamps[-1])]
        fig.add_trace(go.Scatter(
            x=w.index.tz_convert("Europe/Dublin"), y=w.to_numpy(),
            name="Wind forecast (EirGrid)",
            line=dict(color=WIND_COLOUR, width=2),
            hovertemplate="%{y:,.0f} MW<extra>Wind forecast</extra>"),
            row=2, col=1)
        fig.update_yaxes(title_text="Wind (MW)", row=2, col=1)

    x0 = block[0].tz_convert("Europe/Dublin")
    x1 = (block[-1] + pd.Timedelta("15min")).tz_convert("Europe/Dublin")
    fig.add_vrect(x0=x0, x1=x1, fillcolor=WINDOW_COLOUR, opacity=0.14,
                  line_width=0, row=1, col=1,
                  annotation_text="cleanest 3 h",
                  annotation_position="top left",
                  annotation_font_color=WINDOW_COLOUR)
    if rows == 2:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=WINDOW_COLOUR,
                      opacity=0.14, line_width=0, row=2, col=1)
    fig.add_vline(x=origin.tz_convert("Europe/Dublin"),
                  line=dict(color="#8a8f8a", width=1, dash="dot"),
                  row="all", col=1)

    fig.update_yaxes(title_text="g CO2 / kWh", row=1, col=1)
    fig.update_layout(
        height=470 if rows == 2 else 380,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(size=13))
    fig.update_xaxes(gridcolor="#e8e8e6", zeroline=False)
    fig.update_yaxes(gridcolor="#e8e8e6", zeroline=False)
    return fig


def main() -> None:
    st.set_page_config(page_title="GreenWindow Ireland", page_icon="🍃")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("GreenWindow Ireland")
    st.caption("Cleanest 3-hour window on the Irish grid, next 48 hours. "
               "MSc AI project, National College of Ireland.")

    try:
        history = fetch_ci_history()
    except Exception as exc:  # pragma: no cover - network path
        st.error(f"Could not reach the EirGrid dashboard API: {exc}")
        st.stop()

    result = student_forecast(history)
    demo_mode = result is None
    if demo_mode:
        forecast, model_meta = naive_forecast(history), None
        st.warning("DEMO mode: student model assets not bundled yet; "
                   "showing a seasonal-naive forecast.")
    else:
        forecast, model_meta = result

    try:
        wind = fetch_series("wind", "windforecast", 1, 3)
    except Exception:  # pragma: no cover - optional series
        wind = None

    # The EirGrid feed can lag by hours; schedule only from wall-clock
    # now so the recommendation can never sit in the past.
    origin = history.dropna().index[-1]
    now = pd.Timestamp.now(tz="UTC")
    elapsed = max(0, int((now - origin) / pd.Timedelta("15min")))
    if elapsed > HORIZON - 2 * BLOCK_STEPS:
        st.error("EirGrid's feed is too far behind right now to schedule "
                 "ahead; try again later.")
        st.stop()
    live = forecast[elapsed:]
    start = elapsed + choose_block(live, window=len(live))
    stamps = pd.date_range(origin, periods=HORIZON + 1, freq="15min",
                           tz="UTC")[1:]
    block = stamps[start:start + BLOCK_STEPS]
    now_ci = float(np.mean(forecast[elapsed:elapsed + BLOCK_STEPS]))
    win_ci = float(np.mean(forecast[start:start + BLOCK_STEPS]))

    # What is being charged?
    choice = st.selectbox("What do you want to run?", list(LOADS),
                          index=0)
    kwh = LOADS[choice]
    if kwh is None:
        kwh = st.number_input("Energy needed (kWh)", min_value=0.5,
                              max_value=100.0, value=60.0, step=0.5)
    saved_g = (now_ci - win_ci) * kwh

    # Hero recommendation card.
    local = block.tz_convert("Europe/Dublin")
    wait = block[0] - now
    wait_h = wait.total_seconds() / 3600
    if start == elapsed or wait_h <= 0.25:
        chip = "the grid is at its cleanest now, plug in"
    else:
        chip = (f"starts in about {wait_h:.0f} h"
                if wait_h >= 1 else "starts within the hour")
    st.markdown(f"""
<div class="gw-hero">
  <div class="gw-kicker">Recommended window</div>
  <p class="gw-window">{local[0]:%a %H:%M} to
     {(local[-1] + pd.Timedelta("15min")):%H:%M}</p>
  <div class="gw-sub">Charging then instead of now saves an estimated
     <b>{fmt_saving(saved_g)}</b> for a {kwh:g} kWh load.</div>
  <span class="gw-chip">{chip}</span>
</div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Window forecast", f"{win_ci:.0f} g/kWh")
    col2.metric("Charging right now", f"{now_ci:.0f} g/kWh",
                delta=f"{win_ci - now_ci:+.0f} in window",
                delta_color="inverse")
    col3.metric("Estimated saving", fmt_saving(saved_g))

    forecast_name = ("Student forecast" if not demo_mode
                     else "Seasonal-naive (DEMO)")
    st.plotly_chart(
        build_chart(history, forecast, stamps, block, origin, wind,
                    forecast_name),
        use_container_width=True)

    # Why this window?
    st.subheader("Why this window?")
    pct = 100.0 * (now_ci - win_ci) / now_ci if now_ci > 0 else 0.0
    if start == elapsed or pct < 1:
        st.write("The forecast is fairly flat over the next 48 hours, so "
                 "there is little to gain from waiting. The recommended "
                 "block is simply the cheapest one available.")
    else:
        st.write(f"The model expects carbon intensity to fall roughly "
                 f"{pct:.0f}% by {local[0]:%A %H:%M}, from about "
                 f"{now_ci:.0f} to {win_ci:.0f} g/kWh.")
    if wind is not None:
        w_win = float(wind.reindex(block).mean())
        w_day = float(wind.reindex(stamps[elapsed:elapsed + 96]).mean())
        if np.isfinite(w_win) and np.isfinite(w_day) and w_day > 0:
            rel = 100.0 * (w_win - w_day) / w_day
            base = (f"EirGrid's own wind forecast averages "
                    f"{w_win:,.0f} MW during the window, about "
                    f"{abs(rel):.0f}% "
                    f"{'above' if rel >= 0 else 'below'} the coming "
                    f"day's average of {w_day:,.0f} MW.")
            if rel >= 5:
                tail = (" That extra wind displaces gas and coal, which "
                        "is what pulls the carbon intensity down.")
            elif rel <= -5:
                tail = (" So this window is less about a wind peak and "
                        "more about low demand: fewer fossil plants "
                        "need to run at that hour.")
            else:
                tail = (" Wind and demand together set how much fossil "
                        "generation is on the grid at any hour.")
            st.write(base + tail)

    # Model card.
    with st.expander("About the model"):
        if model_meta is not None:
            st.write(f"The forecast comes from a distilled student: a "
                     f"{model_meta['params']:,}-parameter PatchTST "
                     f"trained to imitate a fine-tuned Chronos-Bolt "
                     f"Small teacher (about 48 million parameters) on "
                     f"Irish grid data from 2020 to 2025.")
            st.write(f"In the project benchmark the student matches its "
                     f"teacher's 24-hour accuracy (MASE 0.57 to 0.58 for "
                     f"both) at roughly 3% of the size. This page's "
                     f"forecast ran on CPU in "
                     f"{model_meta['latency_ms']:.0f} ms; the teacher "
                     f"needs about 115 ms per forecast on the same "
                     f"benchmark hardware.")
        else:
            st.write("DEMO mode uses a seasonal-naive forecast (yesterday "
                     "repeated). The distilled student model is bundled "
                     "for the full experience.")
        st.write("The recommendation logic is the same 3-hour contiguous "
                 "block scheduler evaluated in the report's RQ2: a spiky "
                 "forecast cannot harvest scattered dips, so the window "
                 "you see is one you could actually use.")

    stale_note = (" EirGrid's feed is running behind; forecasts beyond "
                  "24 h lead carry more error." if elapsed > 12 else "")
    st.caption(f"Data: EirGrid smart grid dashboard, 15-minute carbon "
               f"intensity for the all-island grid. Latest observation "
               f"{origin.tz_convert('Europe/Dublin'):%d %b %H:%M} local; "
               f"data refreshes every 15 minutes.{stale_note}")


if __name__ == "__main__":
    main()
