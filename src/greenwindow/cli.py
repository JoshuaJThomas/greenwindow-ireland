"""Command line: `greenwindow forecast | export-onnx | serve`."""
import argparse
import json
import pathlib
import sys

import pandas as pd


def cmd_forecast(args: argparse.Namespace) -> None:
    from .eirgrid import fetch_ci_history
    from .forecast import Forecaster, summarise

    forecaster = Forecaster.from_assets(args.assets)
    history = fetch_ci_history()
    fc = forecaster.predict(history)
    out = summarise(fc, history, forecaster.n_params,
                    now=pd.Timestamp.now(tz="UTC"))
    if args.json:
        json.dump(out, sys.stdout)
        print()
        return
    w = out["window"]
    start = pd.Timestamp(w["start"]).tz_convert("Europe/Dublin")
    end = pd.Timestamp(w["end"]).tz_convert("Europe/Dublin")
    last = fc.origin.tz_convert("Europe/Dublin")
    print(f"Last EirGrid reading: {last:%a %d %b %H:%M}")
    print(f"Cleanest 3 hours:     {start:%a %H:%M}-{end:%H:%M}, "
          f"{w['mean_g_per_kwh']:.0f} g CO2/kWh "
          f"({w['saving_vs_now_pct']:.0f}% below running now)")
    print(f"Model: {out['model_params']:,} parameters, "
          f"{out['latency_ms']:.0f} ms on CPU")


def cmd_export(args: argparse.Namespace) -> None:
    from .forecast import Forecaster
    from .onnx_export import export_onnx

    path = export_onnx(Forecaster.from_assets(args.assets), args.out)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    uvicorn.run("greenwindow.api:app", host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="greenwindow")
    ap.add_argument("--assets", type=pathlib.Path, default=None,
                    help="folder with student.pt and stats.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forecast", help="live 48 h forecast and best window")
    f.add_argument("--json", action="store_true")
    f.set_defaults(func=cmd_forecast)
    e = sub.add_parser("export-onnx", help="write the student as ONNX")
    e.add_argument("--out", type=pathlib.Path,
                   default=pathlib.Path("assets/student.onnx"))
    e.set_defaults(func=cmd_export)
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
