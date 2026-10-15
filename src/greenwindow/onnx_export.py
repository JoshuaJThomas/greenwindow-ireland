"""Export the student to ONNX and run it with ONNX Runtime.

ONNX Runtime is a ~15 MB dependency against PyTorch's ~200 MB, so a
deployment that only forecasts can drop torch altogether. The test suite
checks the exported graph against the PyTorch model on real EirGrid data.
"""
import pathlib
import time

import numpy as np
import pandas as pd
import torch

from .forecast import Forecast, Forecaster


def export_onnx(forecaster: Forecaster, path: str | pathlib.Path,
                opset: int = 17) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Batch is fixed at 1: the exporter can't trace Unfold (the patching
    # step) with a symbolic batch size, and the service forecasts one grid.
    example = torch.zeros(1, forecaster.lookback, len(forecaster.columns))
    torch.onnx.export(
        forecaster.model, (example,), str(path),
        input_names=["x"], output_names=["y"],
        opset_version=opset, dynamo=False)
    return path


class OnnxForecaster:
    """Same interface as Forecaster.predict, backed by ONNX Runtime."""

    def __init__(self, path: str | pathlib.Path, forecaster: Forecaster):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1      # match a small CPU runner
        self.session = ort.InferenceSession(
            str(path), opts, providers=["CPUExecutionProvider"])
        self.base = forecaster              # features and scaling

    def predict(self, history: pd.Series) -> Forecast:
        x = self.base.prepare(history)
        t0 = time.perf_counter()
        (out,) = self.session.run(["y"], {"x": x})
        latency = (time.perf_counter() - t0) * 1000.0
        return Forecast(origin=history.dropna().index[-1],
                        values=self.base.to_grams(np.asarray(out)[0, :, 0]),
                        latency_ms=latency)
