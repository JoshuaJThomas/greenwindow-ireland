import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from greenwindow.onnx_export import OnnxForecaster, export_onnx  # noqa: E402


@pytest.fixture(scope="module")
def onnx_model(forecaster, tmp_path_factory):
    path = export_onnx(forecaster, tmp_path_factory.mktemp("onnx") / "student.onnx")
    return OnnxForecaster(path, forecaster)


def test_onnx_matches_pytorch_on_real_data(forecaster, onnx_model, history):
    torch_fc = forecaster.predict(history)
    onnx_fc = onnx_model.predict(history)
    # within 0.05 g CO2/kWh at every one of the 192 steps
    assert np.abs(torch_fc.values - onnx_fc.values).max() < 0.05
    assert onnx_fc.origin == torch_fc.origin


def test_onnx_picks_the_same_window(forecaster, onnx_model, history):
    assert onnx_model.predict(history).best_window()[0] == \
        forecaster.predict(history).best_window()[0]
